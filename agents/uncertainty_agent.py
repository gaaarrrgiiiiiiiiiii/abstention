import numpy as np

class UncertaintyAgent:
    """
    The Uncertainty Agent fuses multiple sources of uncertainty:
    1. Base Model Confidence (Aleatoric)
    2. MC Dropout / Deep Ensemble Disagreement (Epistemic)
    3. LLM Verbalized Confidence (Contextual)
    """
    
    def __init__(self, use_llm=True):
        self.use_llm = use_llm
        if self.use_llm:
            try:
                from transformers import pipeline
                print("Uncertainty Agent: Initializing LLM pipeline for verbalized confidence (this may take a moment)...")
                # Using a very small model for demonstration. In production, use Phi-3 or similar.
                self.llm = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", device_map="auto")
            except ImportError:
                print("Warning: transformers library not found. Falling back to template-based confidence.")
                self.use_llm = False
            except Exception as e:
                print(f"Warning: Could not load LLM ({e}). Falling back to template.")
                self.use_llm = False

    def get_verbalized_confidence(self, features_dict, base_prob, epistemic_unc):
        """
        Queries an LLM (or heuristic fallback) for a verbalized confidence score
        indicating how confident the model should be about this transaction.

        Returns a float in [0, 1] where 1 = very confident, 0 = very uncertain.
        """
        if not self.use_llm:
            # Improved heuristic fallback (replaces the constant 0.5 placeholder).
            # High confidence = base_prob far from 0.5 AND low epistemic uncertainty.
            # Use a sigmoid-like function of distance from the decision boundary.
            margin = abs(base_prob - 0.5)                     # 0..0.5
            margin_score = 1.0 - np.exp(-6 * margin)          # sigmoid-like, 0..1
            epi_penalty  = np.clip(epistemic_unc * 4, 0, 1)   # penalise high epistemic unc
            confidence   = float(np.clip(margin_score - 0.5 * epi_penalty, 0.05, 0.95))
            return confidence

        # Prepare a prompt
        prompt = f"""<|system|>
You are a fraud detection expert. Assess the confidence of fraud based on the following metrics.
Return ONLY a float between 0.0 and 1.0 representing your confidence.
<|user|>
Transaction Features: {features_dict}
Model Fraud Probability: {base_prob:.4f}
Model Epistemic Uncertainty: {epistemic_unc:.4f}
What is your confidence score?
<|assistant|>
"""
        try:
            # Generate response
            response = self.llm(prompt, max_new_tokens=10, return_full_text=False)[0]['generated_text']
            # Extract float from response
            import re
            match = re.search(r"0\.\d+|1\.0", response)
            if match:
                return float(match.group())
            else:
                return 0.5 # Default if parsing fails
        except Exception as e:
            print(f"LLM Generation Error: {e}")
            return 0.5

    def fuse_uncertainty(self, base_probs, epistemic_unc, features_list=None):
        """
        Fuses the different uncertainty signals into a single composite uncertainty score.
        Higher score = More uncertain.
        
        Args:
            base_probs (np.ndarray): Softmax probabilities from the base model (N, num_classes).
            epistemic_unc (np.ndarray): Disagreement/variance from ensemble or MC dropout (N,).
            features_list (list of dicts): Original transaction features for LLM context.
            
        Returns:
            composite_uncertainty (np.ndarray): Fused uncertainty score (N,).
        """
        N = len(base_probs)
        composite_unc = np.zeros(N)
        
        for i in range(N):
            # Aleatoric uncertainty (Entropy of base prediction)
            prob = base_probs[i]
            # Clip to avoid log(0)
            prob_clipped = np.clip(prob, 1e-7, 1.0)
            aleatoric_unc = -np.sum(prob_clipped * np.log(prob_clipped))
            
            # Normalize aleatoric (max entropy for binary classification is log(2) ~ 0.693)
            aleatoric_unc = aleatoric_unc / np.log(2)
            
            epi_u = epistemic_unc[i]
            
            if self.use_llm and features_list is not None:
                verb_conf = self.get_verbalized_confidence(features_list[i], prob[1], epi_u)
                verb_unc = 1.0 - verb_conf
            else:
                verb_unc = 0.5
                
            # Fusion: weighted average (aleatoric, epistemic, verbalized)
            # Weights reflect reliability of each source:
            #   aleatoric  — direct from model softmax, most reliable
            #   epistemic   — ensemble disagreement, reliable but costly
            #   verbalized  — heuristic/LLM, less reliable
            composite_unc[i] = 0.45 * aleatoric_unc + 0.40 * epi_u + 0.15 * verb_unc
            
        return composite_unc
