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
        Queries an LLM to state its confidence on whether the transaction is fraudulent.
        """
        if not self.use_llm:
            # Fallback heuristic
            if base_prob > 0.8 and epistemic_unc < 0.05:
                return 0.9 # High confidence
            elif base_prob < 0.2 and epistemic_unc < 0.05:
                return 0.9 # High confidence
            else:
                return 0.4 # Low confidence

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
                
            # Fusion: simple weighted average (weights can be learned later via gating network)
            # w1: aleatoric, w2: epistemic, w3: verbalized
            composite_unc[i] = 0.4 * aleatoric_unc + 0.4 * epi_u + 0.2 * verb_unc
            
        return composite_unc
