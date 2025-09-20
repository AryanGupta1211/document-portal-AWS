
import os
from typing import List, Tuple, Optional, Dict, Any
from langchain_ollama import OllamaLLM
import ollama
import pandas as pd

# DeepEval
from deepeval.synthesizer import Synthesizer
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.test_case import LLMTestCase
from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)

from langchain_community.vectorstores import FAISS
from groq import Groq
from utils.model_loader import ModelLoader
from utils.deepeval_evaluation.custom_llm import OllamaDeepEvalLLM, GroqDeepEvalLLM, OpenRouterDeepEvalLLM
from dotenv import load_dotenv

class Evaluater:
    def __init__(
        self,
        faiss_folder: str = "faiss_index",
        index_name: str = "index",
        ollama_model_for_generation: str = "llama3.1:latest",
        ollama_model_for_synthesizer: Optional[str] = None,
        retriever_k: int = 3,
    ):
        load_dotenv()
        self.client = Groq()
        self.confident_api_key = os.getenv("CONFIDENT_API_KEY", None)
        self.model_loader = ModelLoader()
        self.embedding_model = self.model_loader.load_embeddings()
        self.vectorestore = FAISS.load_local(
            faiss_folder,
            self.embedding_model,
            index_name,
            allow_dangerous_deserialization=True,
        )
        # self.llm = OllamaDeepEvalLLM(model=ollama_model_for_generation)
        self.llm = GroqDeepEvalLLM(model="llama-3.3-70b-versatile")
        # self.llm = OpenRouterDeepEvalLLM(model="openai/gpt-oss-120b:free")
        self.ollama_client = ollama
        self.retriever = self.vectorestore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": retriever_k},
        )
        synth_llm = OllamaLLM(model=ollama_model_for_synthesizer or ollama_model_for_generation)
        try:
            self.synthesizer = Synthesizer(model=self.llm)
        except TypeError:
            self.synthesizer = Synthesizer()
            self._synthesizer_llm_fallback = synth_llm

        self.default_metrics = [
            AnswerRelevancyMetric(threshold=0.7, model=self.llm, async_mode=False),
            # ContextualPrecisionMetric(threshold=0.7, model=self.llm),
            # ContextualRecallMetric(threshold=0.6, model=self.llm),
            # ContextualRelevancyMetric(threshold=0.6, model=self.llm, async_mode=False),
            # FaithfulnessMetric(threshold=0.6, model=self.llm, async_mode=False),
        ]
        self.queries = [
            "What is the Transformer model?",
            "How does self-attention work?",
            "What is scaled dot-product attention?",
            "How does the Transformer handle parallelization?",
            "What optimizer was used during training?",
            "What regularization techniques were applied?"
        ]
        
    # contexts_list: List[List[str]],
    def generate_goldens_from_contexts(self,  max_goldens_per_context: int = 2) -> EvaluationDataset:
        """
        Generate goldens for all the set of queries.
        """
        contexts_list= []
        for query in self.queries:
            docs = self.retriever.invoke(query)
            contexts = [d.page_content for d in docs]
            contexts_list.append(contexts)
        try:
            goldens_raw = self.synthesizer.generate_goldens_from_contexts(
                contexts=contexts_list,
                include_expected_output=True,
                max_goldens_per_context=max_goldens_per_context,
            )
            print(f"==============raw goldens=========================\n {goldens_raw}")
            print(f"==============pandas goldens======================= \n {pd.DataFrame(goldens_raw)}")
        except Exception as e:
            raise RuntimeError(f"Synthesizer failed to generate goldens: {e}\n")

        # Normalize into Golden objects and wrap into EvaluationDataset
        golden_objs: List[Golden] = []
        for g in goldens_raw:
            golden_objs.append(
            Golden(
                input=g.input or g.query or "",
                expected_output=getattr(g, "expected_output", None),
                metadata=getattr(g, "metadata", None),
                )
            )
        dataset = EvaluationDataset(goldens=golden_objs)
        return dataset

    def upload_dataset_to_confident(self, dataset: EvaluationDataset, name: str):
        """
        Uploads EvaluationDataset to Confident AI (DeepEval backend). Requires CONFIDENT_API_KEY in env.
        """
        if not self.confident_api_key:
            raise EnvironmentError("CONFIDENT_API_KEY not found in environment. Set it to upload datasets.")
        dataset.push(alias=name)
        print(f"Uploaded dataset '{name}' to Confident AI.")
    
    def rag_pipeline(self, query: str) -> Tuple[str, List[str]]:
        
        docs = self.retriever.invoke(query)
        contexts = [d.page_content for d in docs]

        # Build a compact prompt — you can refine the system prompt or temperature as needed
        context_block = "\n\n---\n\n".join(contexts) if contexts else ""
        prompt = (
            "You are a helpful assistant. Use only the following context to answer the question.\n\n"
            f"{context_block}\n\nQuestion: {query}\n\nAnswer succinctly and cite which context you used."
        )

        response = OllamaLLM(
                model="llama3.1:latest",
                temperature=0.3
            ).invoke(prompt)
        
        return response
        
        # chat_completion = self.client.chat.completions.create(
        #     messages=[
        #         {
        #             "role": "user",
        #             "content": prompt,
        #         }
        #     ],
        #     model="llama-3.3-70b-versatile",
        # )
        
        # response_text = chat_completion.choices[0].message.content
        # return response_text
        
    
    def retrieved_contexts(self, query: str) -> List[str]:
        docs = self.retriever.invoke(query)
        contexts = [d.page_content for d in docs]
        return contexts
    
    def evaluate_and_upload(self, dataset: EvaluationDataset):
        test_cases = []
        for golden in dataset.goldens[:5]:
            test_case = LLMTestCase(
                input=golden.input,
                actual_output=self.rag_pipeline(golden.input),
                retrieval_context=self.retrieved_contexts(golden.input),
                expected_output=golden.expected_output,
            )
            dataset.add_test_case(test_case)
            test_cases.append(test_case)

        # Run evaluation locally
        results = evaluate(test_cases, metrics=self.default_metrics)
        print(results)

        dataset.push(alias="transformer-faqs")
        print(f"Uploaded test cases to dataset 'transformer-faqs'")

    
    # def evaluate_query_live(
    #     self,
    #     query: str,
    #     metrics: Optional[List[Any]] = None,
    #     max_goldens_per_context: int = 1,
    #     upload_goldens: bool = False,
    #     upload_name_prefix: str = "live-golden",
    # ) -> Dict[str, Any]:
    #     """
    #     Perform real-time evaluation for a single incoming query.

    #     Steps:
    #      1. Retrieve top-k contexts for the query
    #      2. Synthesize golden Q/A(s) from those contexts (using Synthesizer/Ollama)
    #      3. Run your RAG pipeline to generate an actual answer (Ollama)
    #      4. Create LLMTestCase(s) and run evaluate(...) on them with chosen metrics
    #      5. Return a dict with full results, including goldens, actual_output, and metric results

    #     Returns:
    #         {
    #             "query": query,
    #             "contexts": [...],
    #             "generated_goldens": [...],
    #             "actual_output": "...",
    #             "results": {...}  # DeepEval results/metrics
    #         }
    #     """
    #     metrics = metrics or self.default_metrics

    #     # 1) Retrieve contexts
    #     docs = self.retriever.invoke(query)
    #     contexts = [d.page_content for d in docs]

    #     # 2) Generate goldens from these contexts
    #     dataset = self.generate_goldens_from_contexts([contexts], max_goldens_per_context=max_goldens_per_context)

    #     # Optionally upload goldens immediately
    #     if upload_goldens:
    #         try:
    #             # Make upload name unique with query snippet + timestamp
    #             import time, hashlib
    #             token = hashlib.sha1((query + str(time.time())).encode()).hexdigest()[:8]
    #             name = f"{upload_name_prefix}-{token}"
    #             self.upload_dataset_to_confident(dataset, name=name)
    #         except Exception as e:
    #             print("Warning: upload to Confident AI failed:", e)

    #     # 3) Run RAG pipeline to get actual output and contexts (we already have contexts, but call pipeline for consistent behavior)
    #     actual_output, retrieved_contexts = self.rag_pipeline(query)

    #     # 4) Build test cases from goldens and run evaluate()
    #     # Use dataset.evals_iterator() to iterate over goldens
    #     test_cases = []
    #     for golden in dataset.evals_iterator():
    #         tc = LLMTestCase(
    #             input=golden.input,
    #             actual_output=actual_output,
    #             retrieval_context=retrieved_contexts,
    #             expected_output=golden.expected_output,
    #         )
    #         test_cases.append(tc)

    #     # Evaluate all test_cases at once (DeepEval evaluate supports list of LLMTestCase)
    #     results = evaluate(test_cases, metrics=metrics)

    #     # 5) return a structured result
    #     return {
    #         "query": query,
    #         "contexts": contexts,
    #         "generated_goldens": [ {"input": g.input, "expected_output": g.expected_output, "metadata": g.metadata} for g in dataset.evals_iterator() ],
    #         "actual_output": actual_output,
    #         "results": results,
    #     }

if __name__ == "__main__":
    ev = Evaluater(
        faiss_folder="faiss_index",
        index_name="index",
        ollama_model_for_generation="llama3.1:latest",
    )
    # Pull or generate dataset
    dataset = EvaluationDataset()
    dataset.pull(alias="transformer-faqs")
    ev.evaluate_and_upload(dataset)
    
    # datasets = ev.generate_goldens_from_contexts()
    # ev.upload_dataset_to_confident(datasets, name="transformer-faqs")
    
    # transformer_queries = [
    #     "What is the Transformer model?",
    #     "How does self-attention work?",
    #     "What are the advantages of Transformer over RNNs?",
    #     "What is multi-head attention?",
    #     "How is positional encoding implemented?",
    #     "What tasks was the Transformer tested on?",
    #     "What was the BLEU score for English-German translation?",
    #     "How many layers are in the encoder and decoder?",
    #     "What is scaled dot-product attention?",
    #     "How does the Transformer handle parallelization?",
    #     "What optimizer was used during training?",
    #     "What regularization techniques were applied?",
    #     "What is the purpose of masking in the decoder?",
    #     "How does the Transformer compare to ByteNet and ConvS2S?",
    #     "What are the dimensions of the model (d_model, d_ff)?",
    #     "How many attention heads were used?",
    #     "What dataset was used for training?",
    #     "How long did it take to train the big model?",
    #     "What is label smoothing and why was it used?",
    #     "How does the Transformer compute output probabilities?"
    # ]
