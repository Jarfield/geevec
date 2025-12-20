"""Multi-stage orchestration for AILA statute scenario generation."""

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterable, List, Optional

# 将 data_augmentation 的 code 目录加入系统路径，复用公共 LLM 与生成器逻辑
CURRENT_DIR = os.path.dirname(__file__)
PARENT_CODE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "code"))
if PARENT_CODE_DIR not in sys.path:
    sys.path.append(PARENT_CODE_DIR)

from utils import clean_content  # type: ignore
from llm import LLM  # type: ignore
from triplet_generator import TripletGenerator  # type: ignore
from doc_synthesis_generator import DocSynthesisGenerator  # type: ignore

from .aila_config import AILAFieldMapping
from .aila_utils import structural_chunker
from .aila_prompts import get_layman_summary_prompt, get_syllogism_scenario_prompt


class AILATripletGenerator(TripletGenerator):
    """Triplet generator with AILA-specific syllogism prompting."""

    def generate_triplets(
        self,
        data: dict,
        task,
        examples_pool=None,
        num_examples: int = 3,
        num_variants_per_doc: int = 1,
        narrative_focus: Optional[str] = None,
        debug_mode: bool = False,
        **kwargs,
    ):
        statute_text = data.get("text", "")
        layman_summary = data.get("layman_summary")

        result_list: List[Dict[str, str]] = []
        for _ in range(num_variants_per_doc):
            prompt = get_syllogism_scenario_prompt(
                statute=statute_text,
                layman_summary=layman_summary,
            )
            scenario = self.chat(prompt, **kwargs)[0]
            cleaned = clean_content(scenario)
            payload: Dict[str, str] = {
                "statute_id": data.get("_id"),
                "statute_title": data.get("title"),
                "statute_text": statute_text,
                "layman_summary": layman_summary,
                "base_scenario": cleaned,
            }
            if debug_mode:
                payload["generation_prompt"] = prompt
            result_list.append(payload)

        return result_list


class AILAPipeline:
    """Pipeline that stitches together layman summaries, scenarios, evolutions and near-miss negatives."""

    def __init__(
        self,
        model: str = "Qwen2-5-Coder-32B-Instruct",
        model_type: str = "open-source",
        port: int = 8000,
        thread_count: int = 8,
    ):
        self.thread_count = thread_count
        self.doc_synth = DocSynthesisGenerator(model=model, model_type=model_type, port=port)
        self.triplet_generator = AILATripletGenerator(model=model, model_type=model_type, port=port)
        self.llm = LLM(model=model, model_type=model_type, port=port)
        self.fields = AILAFieldMapping()

    # -------------------- Stage 0: preprocessing -------------------- #
    def chunk_corpus(self, docs: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
        """Split IPC corpus entries into structured statute chunks."""

        chunks: List[Dict[str, str]] = []
        for doc in docs:
            doc_id = doc.get(self.fields.id_key) or ""
            title = doc.get(self.fields.title_key) or ""
            text = doc.get(self.fields.text_key) or ""
            for idx, chunk_text in enumerate(structural_chunker(text)):
                chunks.append(
                    {
                        "_id": f"{doc_id}-sec-{idx:03d}",
                        "title": title,
                        "text": chunk_text,
                        "source_id": doc_id,
                    }
                )
        return chunks

    # -------------------- Stage 1: layman summary -------------------- #
    def generate_layman_summaries(
        self,
        statute_chunks: List[Dict[str, str]],
        **llm_kwargs,
    ) -> List[Dict[str, str]]:
        """Bridge statutes to plain-language summaries via DocSynthesisGenerator."""

        seeds = [{"_id": chunk["_id"], "text": chunk["text"]} for chunk in statute_chunks]
        llm_kwargs.setdefault("thread_count", self.thread_count)
        layman_results = self.doc_synth.run(
            seed_docs=seeds,
            task_type="ailastatutes",
            language="en",
            num_variants_per_seed=1,
            thread_count=llm_kwargs.pop("thread_count"),
            **llm_kwargs,
        )

        summary_lookup = {item["seed_id"]: item for item in layman_results}
        enriched: List[Dict[str, str]] = []
        for chunk in statute_chunks:
            summary_obj = summary_lookup.get(chunk["_id"])
            layman_summary = summary_obj["text"] if summary_obj else ""

            # Fallback: force a plain-language rewrite using the dedicated prompt.
            if not layman_summary:
                fallback_prompt = get_layman_summary_prompt(chunk["text"])
                layman_summary = clean_content(self.llm.chat(fallback_prompt, **llm_kwargs)[0])

            enriched.append({**chunk, "layman_summary": layman_summary})
        return enriched

    # -------------------- Stage 2: base scenario generation -------------------- #
    def generate_base_scenarios(
        self,
        layman_chunks: List[Dict[str, str]],
        **llm_kwargs,
    ) -> List[Dict[str, str]]:
        """Generate S0 scenarios using the reverse syllogism prompt."""

        llm_kwargs.setdefault("thread_count", self.thread_count)
        scenarios = self.triplet_generator.run(
            positives=layman_chunks,
            task_type="ailastatutes",
            language="en",
            thread_count=llm_kwargs.pop("thread_count"),
            debug_mode=llm_kwargs.pop("debug_mode", False),
            **llm_kwargs,
        )
        return scenarios

    # -------------------- Stage 3: legal evolution -------------------- #
    def evolve_scenarios(
        self,
        base_cases: List[Dict[str, str]],
        rounds: int = 2,
        **llm_kwargs,
    ) -> List[Dict[str, str]]:
        """Evolve scenarios by injecting distractors and softening intent cues."""

        llm_kwargs.setdefault("temperature", 0.9)
        llm_kwargs.setdefault("top_p", 0.9)

        def _evolve(case: Dict[str, str]):
            prompt = f"""\
You will evolve a legal fact pattern to create noisy training variants.
- Start from the base scenario below (S0).
- Inject irrelevant distractors such as weather, background sounds, or bystander chatter.
- Blur intentionality: downplay explicit malicious intent, use ambiguous phrasing for motives.
- Keep the core actions intact so the statute remains tentatively applicable.
- Generate {rounds} variants, each as a short paragraph.

S0:
{case.get('base_scenario')}

Return {rounds} numbered variants."""
            raw = self.llm.chat(prompt, **llm_kwargs)[0]
            cleaned = clean_content(raw)
            case_with_evol = dict(case)
            case_with_evol["evolutions"] = cleaned
            return case_with_evol

        with ThreadPoolExecutor(max_workers=self.thread_count) as executor:
            return list(executor.map(_evolve, base_cases))

    # -------------------- Stage 4: near-miss negatives -------------------- #
    def generate_near_miss_negatives(
        self,
        evolved_cases: List[Dict[str, str]],
        confusing_statutes: Optional[List[str]] = None,
        **llm_kwargs,
    ) -> List[Dict[str, str]]:
        """Create hard negatives that resemble but do not fully satisfy the statute."""

        confusing_statutes = confusing_statutes or ["IPC 299 (culpable homicide)", "IPC 300 (murder)"]
        llm_kwargs.setdefault("temperature", 0.8)
        llm_kwargs.setdefault("top_p", 0.9)

        def _near_miss(case: Dict[str, str]):
            prompt = f"""\
Craft a near-miss scenario based on the statute application below. The output should look plausible
but deliberately miss at least one key element, making it a hard negative example.
- Base scenario (possibly evolved):
{case.get('evolutions') or case.get('base_scenario')}
- Similar statutes to contrast against: {', '.join(confusing_statutes)}
- Include subtle distractors (e.g., provocation, chaotic crowd) to mask the missing element.

Return one concise paragraph in English."""
            raw = self.llm.chat(prompt, **llm_kwargs)[0]
            cleaned = clean_content(raw)
            case_with_neg = dict(case)
            case_with_neg["near_miss_negative"] = cleaned
            return case_with_neg

        with ThreadPoolExecutor(max_workers=self.thread_count) as executor:
            return list(executor.map(_near_miss, evolved_cases))


__all__ = ["AILAPipeline", "AILATripletGenerator"]
