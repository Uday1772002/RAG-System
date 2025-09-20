"""
RAG Evaluation Module
Evaluates the quality of RAG system responses
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    retrieval_precision: float
    retrieval_recall: float
    retrieval_f1: float
    answer_relevance: float
    citation_accuracy: float
    overall_score: float

class RAGEvaluator:
    """Evaluates RAG system performance"""
    
    def __init__(self):
        self.metrics = {}
        
    def evaluate_retrieval(self, query: str, retrieved_docs: List[Dict], 
                          relevant_docs: List[Dict]) -> Dict[str, float]:
        """
        Evaluate retrieval quality
        
        Args:
            query: Original query
            retrieved_docs: Documents retrieved by the system
            relevant_docs: Ground truth relevant documents
            
        Returns:
            Dictionary with precision, recall, and F1 scores
        """
        try:
            if not retrieved_docs:
                return {
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1': 0.0
                }
                
            # Calculate precision and recall
            relevant_ids = set(doc.get('id') for doc in relevant_docs)
            retrieved_ids = set(doc.get('id') for doc in retrieved_docs)
            
            # True positives: documents that are both retrieved and relevant
            true_positives = len(relevant_ids.intersection(retrieved_ids))
            
            # Precision: TP / (TP + FP)
            precision = true_positives / len(retrieved_ids) if retrieved_ids else 0.0
            
            # Recall: TP / (TP + FN)
            recall = true_positives / len(relevant_ids) if relevant_ids else 0.0
            
            # F1 score: 2 * (precision * recall) / (precision + recall)
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            return {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            
        except Exception as e:
            logger.error(f"Error evaluating retrieval: {e}")
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
            
    def evaluate_answer_relevance(self, query: str, answer: str) -> float:
        """
        Evaluate how relevant the answer is to the query
        
        Args:
            query: Original query
            answer: Generated answer
            
        Returns:
            Relevance score between 0 and 1
        """
        try:
            # Simple keyword overlap-based relevance
            query_words = set(query.lower().split())
            answer_words = set(answer.lower().split())
            
            if not query_words:
                return 0.0
                
            # Jaccard similarity
            intersection = len(query_words.intersection(answer_words))
            union = len(query_words.union(answer_words))
            
            relevance = intersection / union if union > 0 else 0.0
            
            # Boost score if answer is substantial
            if len(answer.strip()) > 50:
                relevance *= 1.1
                
            return min(relevance, 1.0)
            
        except Exception as e:
            logger.error(f"Error evaluating answer relevance: {e}")
            return 0.0
            
    def evaluate_citation_accuracy(self, citations: List[Dict], 
                                 retrieved_docs: List[Dict]) -> float:
        """
        Evaluate the accuracy of citations
        
        Args:
            citations: Generated citations
            retrieved_docs: Retrieved documents
            
        Returns:
            Citation accuracy score between 0 and 1
        """
        try:
            if not citations or not retrieved_docs:
                return 0.0
                
            # Check if citations reference actual retrieved documents
            retrieved_ids = set(doc.get('id') for doc in retrieved_docs)
            citation_ids = set(citation.get('chunk_id') for citation in citations)
            
            # Calculate accuracy as percentage of citations that reference retrieved docs
            accurate_citations = len(citation_ids.intersection(retrieved_ids))
            total_citations = len(citations)
            
            accuracy = accurate_citations / total_citations if total_citations > 0 else 0.0
            
            return accuracy
            
        except Exception as e:
            logger.error(f"Error evaluating citation accuracy: {e}")
            return 0.0
            
    def evaluate_response(self, query: str, answer: str, citations: List[Dict],
                         retrieved_docs: List[Dict], relevant_docs: List[Dict]) -> EvaluationMetrics:
        """
        Comprehensive evaluation of a RAG response
        
        Args:
            query: Original query
            answer: Generated answer
            citations: Citations: Generated citations
            retrieved_docs: Retrieved documents
            relevant_docs: Ground truth relevant documents
            
        Returns:
            EvaluationMetrics object with all scores
        """
        try:
            # Evaluate retrieval
            retrieval_metrics = self.evaluate_retrieval(query, retrieved_docs, relevant_docs)
            
            # Evaluate answer relevance
            answer_relevance = self.evaluate_answer_relevance(query, answer)
            
            # Evaluate citation accuracy
            citation_accuracy = self.evaluate_citation_accuracy(citations, retrieved_docs)
            
            # Calculate overall score (weighted average)
            overall_score = (
                0.4 * retrieval_metrics['f1'] +
                0.3 * answer_relevance +
                0.3 * citation_accuracy
            )
            
            return EvaluationMetrics(
                retrieval_precision=retrieval_metrics['precision'],
                retrieval_recall=retrieval_metrics['recall'],
                retrieval_f1=retrieval_metrics['f1'],
                answer_relevance=answer_relevance,
                citation_accuracy=citation_accuracy,
                overall_score=overall_score
            )
            
        except Exception as e:
            logger.error(f"Error evaluating response: {e}")
            return EvaluationMetrics(
                retrieval_precision=0.0,
                retrieval_recall=0.0,
                retrieval_f1=0.0,
                answer_relevance=0.0,
                citation_accuracy=0.0,
                overall_score=0.0
            )
            
    def generate_evaluation_report(self, evaluations: List[EvaluationMetrics]) -> Dict:
        """
        Generate a comprehensive evaluation report
        
        Args:
            evaluations: List of evaluation results
            
        Returns:
            Dictionary with aggregated metrics
        """
        try:
            if not evaluations:
                return {"error": "No evaluations provided"}
                
            # Aggregate metrics
            total_evaluations = len(evaluations)
            
            avg_precision = np.mean([e.retrieval_precision for e in evaluations])
            avg_recall = np.mean([e.retrieval_precision for e in evaluations])
            avg_f1 = np.mean([e.retrieval_f1 for e in evaluations])
            avg_relevance = np.mean([e.answer_relevance for e in evaluations])
            avg_citation_accuracy = np.mean([e.citation_accuracy for e in evaluations])
            avg_overall = np.mean([e.overall_score for e in evaluations])
            
            # Standard deviations
            std_precision = np.std([e.retrieval_precision for e in evaluations])
            std_recall = np.std([e.retrieval_recall for e in evaluations])
            std_f1 = np.std([e.retrieval_f1 for e in evaluations])
            std_relevance = np.std([e.answer_relevance for e in evaluations])
            std_citation_accuracy = np.std([e.citation_accuracy for e in evaluations])
            std_overall = np.std([e.overall_score for e in evaluations])
            
            return {
                "summary": {
                    "total_evaluations": total_evaluations,
                    "average_overall_score": round(avg_overall, 3)
                },
                "retrieval_metrics": {
                    "precision": {"mean": round(avg_precision, 3), "std": round(std_precision, 3)},
                    "recall": {"mean": round(avg_recall, 3), "std": round(std_recall, 3)},
                    "f1": {"mean": round(avg_f1, 3), "std": round(std_f1, 3)}
                },
                "generation_metrics": {
                    "answer_relevance": {"mean": round(avg_relevance, 3), "std": round(std_relevance, 3)},
                    "answer_relevance": {"mean": round(avg_relevance, 3), "std": round(std_relevance, 3)},
                    "citation_accuracy": {"mean": round(avg_citation_accuracy, 3), "std": round(std_citation_accuracy, 3)}
                },
                "overall_score": {
                    "mean": round(avg_overall, 3),
                    "std": round(std_overall, 3)
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating evaluation report: {e}")
            return {"error": str(e)}
