import os
from typing import List, Tuple, Optional, Dict, Any
from langchain_ollama import OllamaLLM
import ollama
import pandas as pd

# DeepEval - Latest version imports
from deepeval.synthesizer import Synthesizer
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.test_case import LLMTestCase, ConversationalTestCase, Turn
from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    ConversationalGEval,
)
from deepeval.test_case import TurnParams

# Latest tracing imports
from deepeval.tracing import observe, update_current_span
from deepeval.tracing import TraceStatus

from langchain_community.vectorstores import FAISS
from groq import Groq
from utils.model_loader import ModelLoader
from utils.deepeval_evaluation.custom_llm import OllamaDeepEvalLLM, GroqDeepEvalLLM
from dotenv import load_dotenv

class ModernRAGEvaluator:
    
    def __init__(
        self,
        faiss_folder: str = "faiss_index",
        index_name: str = "index",
        ollama_model_for_generation: str = "llama3.1:latest",
        ollama_model_for_synthesizer: Optional[str] = None,
        retriever_k: int = 3,
        metric_collection_name: str = "rag-evaluation-metrics",
        enable_tracing: bool = True,
    ):
        load_dotenv()
        self.client = Groq()
        self.confident_api_key = os.getenv("CONFIDENT_API_KEY", None)
        self.enable_tracing = enable_tracing and self.confident_api_key
        self.metric_collection_name = metric_collection_name
        
        # Initialize models and vector store
        self.model_loader = ModelLoader()
        self.embedding_model = self.model_loader.load_embeddings()
        self.vectorstore = FAISS.load_local(
            faiss_folder,
            self.embedding_model,
            index_name,
            allow_dangerous_deserialization=True,
        )
        
        self.llm = GroqDeepEvalLLM(model="llama-3.3-70b-versatile")
        self.ollama_client = ollama
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": retriever_k},
        )
        
        # Initialize synthesizer
        synth_llm = OllamaLLM(model=ollama_model_for_synthesizer or ollama_model_for_generation)
        try:
            self.synthesizer = Synthesizer(model=self.llm)
        except TypeError:
            self.synthesizer = Synthesizer()
            self._synthesizer_llm_fallback = synth_llm

        # Define metrics following latest patterns
        self.rag_metrics = [
            AnswerRelevancyMetric(threshold=0.8, model=self.llm),
            ContextualPrecisionMetric(threshold=0.8, model=self.llm),
            ContextualRecallMetric(threshold=0.6, model=self.llm),
            ContextualRelevancyMetric(threshold=0.6, model=self.llm),
            FaithfulnessMetric(threshold=0.6, model=self.llm),
        ]
        
        # Separate metrics for component-wise evaluation
        self.retriever_metrics = [
            ContextualRelevancyMetric(threshold=0.6, model=self.llm),
        ]
        
        self.generator_metrics = [
            AnswerRelevancyMetric(threshold=0.8, model=self.llm),
            FaithfulnessMetric(threshold=0.6, model=self.llm),
        ]
        
        # Multi-turn conversation metrics
        self.conversational_faithfulness = ConversationalGEval(
            name="Faithfulness",
            criteria="Determine whether the assistant's responses are factually supported by the retrieved context across the entire conversation.",
            evaluation_params=[TurnParams.ROLE, TurnParams.CONTENT, TurnParams.RETRIEVAL_CONTEXT],
            model=self.llm
        )
        
        self.test_queries = [
            "What is the Transformer model?",
            "How does self-attention work?",
            "What is scaled dot-product attention?",
            "How does the Transformer handle parallelization?",
            "What optimizer was used during training?",
            "What regularization techniques were applied?"
        ]
        
        if self.enable_tracing:
            print(f"✅ Modern RAG evaluation enabled with tracing")
            print(f"📊 Metric collection: {self.metric_collection_name}")
        else:
            print("⚠️ Tracing disabled - set CONFIDENT_API_KEY to enable")

    # Component-wise evaluation following latest patterns
    @observe()
    def retriever_component(self, query: str) -> List[str]:
        """
        Retriever component with tracing.
        """
        docs = self.retriever.invoke(query)
        contexts = [d.page_content for d in docs]
        
        # Update span with retrieval context for evaluation
        update_current_span(
            test_case=LLMTestCase(
                input=query,
                retrieval_context=contexts
            )
        )
        
        return contexts

    @observe()
    def generator_component(self, query: str, text_chunks: List[str]) -> str:
        """
        Generator component with tracing.
        """
        # Build prompt from retrieved chunks
        context_block = "\n\n---\n\n".join(text_chunks) if text_chunks else ""
        prompt = (
            "You are a helpful assistant. Use only the following context to answer the question.\n\n"
            f"{context_block}\n\nQuestion: {query}\n\nAnswer succinctly and cite which context you used."
        )

        # Generate response
        chat_completion = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
        )
        
        response = chat_completion.choices[0].message.content
        
        # Update span with generator output for evaluation
        update_current_span(
            test_case=LLMTestCase(
                input=query,
                actual_output=response,
                retrieval_context=text_chunks
            )
        )
        
        return response

    @observe()
    def rag_pipeline(self, query: str) -> Tuple[str, List[str]]:
        """
        Complete RAG pipeline following latest end-to-end evaluation pattern.
        """
        # Component-wise execution with tracing
        retrieved_contexts = self.retriever_component(query)
        response = self.generator_component(query, retrieved_contexts)
        
        return response, retrieved_contexts

    def evaluate_end_to_end(self, queries: List[str], expected_outputs: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        End-to-end RAG evaluation following latest patterns.
        Creates LLMTestCase for each query and runs comprehensive evaluation.
        """
        if expected_outputs and len(queries) != len(expected_outputs):
            raise ValueError("Number of queries and expected outputs must match")
        
        test_cases = []
        results = []
        
        print(f"🚀 Running end-to-end evaluation on {len(queries)} queries...")
        
        for i, query in enumerate(queries):
            print(f"📝 Processing query {i+1}/{len(queries)}: {query[:50]}...")
            
            # Execute RAG pipeline
            actual_output, retrieved_contexts = self.rag_pipeline(query)
            expected = expected_outputs[i] if expected_outputs else None
            
            # Create test case following latest pattern
            test_case = LLMTestCase(
                input=query,
                actual_output=actual_output,
                retrieval_context=retrieved_contexts,
                expected_output=expected
            )
            
            test_cases.append(test_case)
            results.append({
                "query": query,
                "actual_output": actual_output,
                "expected_output": expected,
                "retrieved_contexts": retrieved_contexts
            })
        
        # Run evaluation with all RAG metrics
        print("🔍 Running evaluation with RAG metrics...")
        evaluation_results = evaluate(test_cases, metrics=self.rag_metrics)
        
        return {
            "test_cases": test_cases,
            "individual_results": results,
            "evaluation_results": evaluation_results,
            "total_queries": len(queries),
            "metrics_used": [metric.__class__.__name__ for metric in self.rag_metrics]
        }

    def evaluate_components_separately(self, queries: List[str]) -> Dict[str, Any]:
        """
        Evaluate retriever and generator as separate components.
        Following: Component-wise evaluation patterns from latest docs.
        """
        print(f"🔧 Running component-wise evaluation on {len(queries)} queries...")
        
        # Create dataset for iteration
        goldens = [Golden(input=query) for query in queries]
        dataset = EvaluationDataset(goldens=goldens)
        
        retriever_results = []
        generator_results = []
        
        # Evaluate components separately using dataset iterator
        for golden in dataset.evals_iterator():
            query = golden.input
            
            # Test retriever component with its metrics
            with observe(metrics=self.retriever_metrics):
                contexts = self.retriever_component(query)
                retriever_results.append({
                    "query": query,
                    "retrieved_contexts": contexts,
                    "context_count": len(contexts)
                })
            
            # Test generator component with its metrics  
            with observe(metrics=self.generator_metrics):
                response = self.generator_component(query, contexts)
                generator_results.append({
                    "query": query,
                    "response": response,
                    "contexts_used": len(contexts)
                })
        
        return {
            "retriever_results": retriever_results,
            "generator_results": generator_results,
            "component_metrics": {
                "retriever": [m.__class__.__name__ for m in self.retriever_metrics],
                "generator": [m.__class__.__name__ for m in self.generator_metrics]
            }
        }

    def evaluate_conversational_rag(self, conversation_turns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Multi-turn RAG evaluation for chatbots and conversational systems.
        """
        print(f"💬 Evaluating conversational RAG with {len(conversation_turns)} turns...")
        
        # Build turns following the latest ConversationalTestCase pattern
        turns = []
        for turn_data in conversation_turns:
            if turn_data["role"] == "user":
                turns.append(Turn(role="user", content=turn_data["content"]))
            else:  # assistant turn
                # For assistant turns, retrieve context if not provided
                retrieval_context = turn_data.get("retrieval_context")
                if not retrieval_context and "content" in turn_data:
                    # Extract query from previous user turn for context retrieval
                    if turns and turns[-1].role == "user":
                        prev_query = turns[-1].content
                        retrieval_context = self.retriever_component(prev_query)
                
                turns.append(Turn(
                    role="assistant",
                    content=turn_data["content"],
                    retrieval_context=retrieval_context or []
                ))
        
        # Create conversational test case
        conv_test_case = ConversationalTestCase(turns=turns)
        
        # Run evaluation with conversational metrics
        evaluation_results = evaluate([conv_test_case], metrics=[self.conversational_faithfulness])
        
        return {
            "conversation_test_case": conv_test_case,
            "evaluation_results": evaluation_results,
            "total_turns": len(turns),
            "user_turns": len([t for t in turns if t.role == "user"]),
            "assistant_turns": len([t for t in turns if t.role == "assistant"])
        }

    def generate_evaluation_dataset(self, max_goldens_per_context: int = 2) -> EvaluationDataset:
        """
        Generate golden dataset from contexts using latest synthesizer patterns.
        """
        print(f"🏗️ Generating evaluation dataset with {len(self.test_queries)} base queries...")
        
        contexts_list = []
        for query in self.test_queries:
            docs = self.retriever.invoke(query)
            contexts = [d.page_content for d in docs]
            contexts_list.append(contexts)
        
        try:
            goldens_raw = self.synthesizer.generate_goldens_from_contexts(
                contexts=contexts_list,
                include_expected_output=True,
                max_goldens_per_context=max_goldens_per_context,
            )
            print(f"✅ Generated {len(goldens_raw)} golden test cases")
        except Exception as e:
            raise RuntimeError(f"Synthesizer failed to generate goldens: {e}")

        # Convert to Golden objects
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
        print(f"📊 Created dataset with {len(golden_objs)} golden test cases")
        return dataset

    def run_comprehensive_evaluation(self, upload_to_confident: bool = True) -> Dict[str, Any]:
        """
        Run comprehensive RAG evaluation covering all aspects.
        """
        print("🚀 Starting comprehensive RAG evaluation...")
        
        results = {}
        
        # 1. End-to-end evaluation
        print("\n=== 1. End-to-End RAG Evaluation ===")
        end_to_end_results = self.evaluate_end_to_end(self.test_queries)
        results["end_to_end"] = end_to_end_results
        
        # 2. Component-wise evaluation
        print("\n=== 2. Component-wise Evaluation ===")
        component_results = self.evaluate_components_separately(self.test_queries[:3])  # Subset for demo
        results["components"] = component_results
        
        # 3. Generate and evaluate with golden dataset
        print("\n=== 3. Golden Dataset Evaluation ===")
        try:
            golden_dataset = self.generate_evaluation_dataset(max_goldens_per_context=1)
            
            # Extract queries from golden dataset
            golden_queries = [golden.input for golden in golden_dataset.goldens]
            golden_expected = [golden.expected_output for golden in golden_dataset.goldens]
            
            # Evaluate against golden dataset
            golden_results = self.evaluate_end_to_end(golden_queries, golden_expected)
            results["golden_dataset"] = golden_results
            
            # Upload dataset to Confident AI
            if upload_to_confident and self.confident_api_key:
                golden_dataset.push(alias="transformer-rag-evaluation")
                print("✅ Uploaded golden dataset to Confident AI")
                
        except Exception as e:
            print(f"❌ Golden dataset evaluation failed: {e}")
            results["golden_dataset_error"] = str(e)
        
        # 4. Sample conversational evaluation
        print("\n=== 4. Conversational RAG Evaluation ===")
        sample_conversation = [
            {"role": "user", "content": "What is the Transformer model?"},
            {"role": "assistant", "content": "The Transformer is a neural network architecture..."},
            {"role": "user", "content": "How does self-attention work in it?"},
            {"role": "assistant", "content": "Self-attention allows the model to..."}
        ]
        
        try:
            conv_results = self.evaluate_conversational_rag(sample_conversation)
            results["conversational"] = conv_results
        except Exception as e:
            print(f"❌ Conversational evaluation failed: {e}")
            results["conversational_error"] = str(e)
        
        print("\n✅ Comprehensive evaluation completed!")
        print(f"📊 Results available on Confident AI dashboard: https://app.confident-ai.com")
        
        return results


def main():
    """
    Demo script showing modern RAG evaluation patterns
    """
    # Initialize evaluator with latest patterns
    evaluator = ModernRAGEvaluator(
        faiss_folder="faiss_index",
        index_name="index",
        ollama_model_for_generation="llama3.1:latest",
        metric_collection_name="modern-rag-metrics",
        enable_tracing=True,
    )
    
    # Run comprehensive evaluation
    results = evaluator.run_comprehensive_evaluation(upload_to_confident=True)
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    
    if "end_to_end" in results:
        print(f"End-to-end queries evaluated: {results['end_to_end']['total_queries']}")
    
    if "components" in results:
        retriever_count = len(results['components']['retriever_results'])
        generator_count = len(results['components']['generator_results'])
        print(f"Component evaluations - Retriever: {retriever_count}, Generator: {generator_count}")
    
    if "golden_dataset" in results:
        golden_count = results['golden_dataset']['total_queries']
        print(f"Golden dataset evaluations: {golden_count}")
    
    if "conversational" in results:
        total_turns = results['conversational']['total_turns']
        print(f"Conversational turns evaluated: {total_turns}")
    
    print(f"\n🌐 View detailed results at: https://app.confident-ai.com")


if __name__ == "__main__":
    main()
