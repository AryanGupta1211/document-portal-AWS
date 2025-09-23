import os
from typing import List, Union
import pandas as pd
from deepeval.test_case import LLMTestCase
from deepeval import evaluate
from deepeval.metrics import (
    ContextualRelevancyMetric,
    AnswerRelevancyMetric,
    FaithfulnessMetric
)
from utils.deepeval_evaluation.custom_llm import GroqDeepEvalLLM
from dotenv import load_dotenv
from logger import GLOBAL_LOGGER as log

class RealtimeRAGEvaluator:

    def __init__(self):
        load_dotenv()
        self.metric_collection_name = "rag-evaluation-metrics"
        self.llm = GroqDeepEvalLLM(model="llama-3.3-70b-versatile")
        
        # # Separate metrics for component-wise evaluation
        # self.rag_metrics = [
        #     RAGASContextualPrecisionMetric(threshold=0.7, model=self.llm),
        #     RAGASAnswerRelevancyMetric(threshold=0.8, model=self.llm),
        #     RAGASFaithfulnessMetric(threshold=0.6, model=self.llm),
        # ]
        # Separate metrics for component-wise evaluation
        self.rag_metrics = [
            ContextualRelevancyMetric(threshold=0.7, model=self.llm),
            AnswerRelevancyMetric(threshold=0.8, model=self.llm),
            FaithfulnessMetric(threshold=0.6, model=self.llm),
        ]

    def evaluate_pipeline(self, query: str, contexts: Union[str, List[str]], actual_output: str):
        """
        Evaluating Real-time RAG pipeline 
        
        Args:
            query: The user's question
            contexts: Either a string of concatenated contexts or a list of context strings
            actual_output: The generated answer
        """
        self.query = query
        self.actual_output = actual_output
        
        if isinstance(contexts, str):
            # If contexts is a single string, split it (assuming it's concatenated with \n\n)
            self.contexts = [ctx.strip() for ctx in contexts.split('\n\n') if ctx.strip()]
        elif isinstance(contexts, list):
            # If contexts is already a list, use it directly
            self.contexts = [str(ctx).strip() for ctx in contexts if str(ctx).strip()]
        else:
            raise ValueError(f"Contexts must be either string or list, got {type(contexts)}")
            
        test_case = LLMTestCase(
            input=self.query,
            retrieval_context=self.contexts, 
            actual_output=self.actual_output
        )
            
        try:
            evaluation_results = evaluate(test_cases=[test_case], metrics=self.rag_metrics)
            
            log.info(
                "RAG evaluation completed successfully",
                query=self.query[:100],
                context_count=len(self.contexts),
                answer_preview=self.actual_output[:100]
            )
            
            return evaluation_results
            
        except Exception as e:
            log.error("Failed to evaluate RAG pipeline", error=str(e))
            raise