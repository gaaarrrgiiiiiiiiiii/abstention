class ReflectionAgent:
    """
    The Reflection Agent provides natural language explanations for *why* 
    the system abstained on a particular transaction.
    This fulfills the "Explainability" and "Human Oversight" requirements of the EU AI Act.
    """
    
    def __init__(self, use_llm=True, llm_model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.use_llm = use_llm
        self.model_name = llm_model_name
        if self.use_llm:
            try:
                from transformers import pipeline
                print(f"Reflection Agent: Initializing LLM pipeline ({self.model_name}) for explanations...")
                # To simulate the ablation, if a larger model is requested, it might require more memory.
                self.llm = pipeline("text-generation", model=self.model_name, device_map="auto")
            except ImportError:
                print("Warning: transformers library not found. Falling back to template-based explanations.")
                self.use_llm = False
            except Exception as e:
                print(f"Warning: Could not load LLM {self.model_name} ({e}). Falling back to template.")
                self.use_llm = False
                
    def generate_explanation(self, features_dict, base_prob, composite_unc):
        """
        Generates an explanation for human reviewers.
        """
        if not self.use_llm:
            return f"System abstained due to high composite uncertainty ({composite_unc:.2f}). Base fraud probability was {base_prob:.2f}."
            
        prompt = f"""<|system|>
You are an AI assistant helping a human fraud analyst. Explain why the AI model was uncertain about the following transaction. Keep it brief.
<|user|>
Transaction Features: {features_dict}
Model Fraud Probability: {base_prob:.4f}
Composite Uncertainty: {composite_unc:.4f}
Why was this transaction flagged for human review (abstained)?
<|assistant|>
"""
        try:
            response = self.llm(prompt, max_new_tokens=50, return_full_text=False)[0]['generated_text']
            return response.strip()
        except Exception as e:
            print(f"LLM Generation Error: {e}")
            return f"System abstained due to high composite uncertainty ({composite_unc:.2f})."
