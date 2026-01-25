"""
Example: Quick Start

Minimal example showing the basic usage of the package.
"""

from hallucination_detection import BatchEnsembleModel, UncertaintyEstimator, HallucinationDetector

# 1. Create BatchEnsemble model
print("Loading model...")
model = BatchEnsembleModel.from_pretrained(
    # "mistralai/Mistral-7B-Instruct-v0.2",
    "google/gemma-2b",
    num_ensemble_members=5
)

# 2. Example prompt
prompt = """[INST] Answer the question based only on the given context.
Context: The Eiffel Tower is located in Paris, France.
Question: What is the capital of France? [/INST]"""

# 3. Compute uncertainty
print("\nComputing uncertainty...")
estimator = UncertaintyEstimator(model)
uncertainties = estimator.compute_uncertainties(prompt, return_tokens=True)

print(f"Predictive Entropy: {uncertainties['predictive_entropy']:.4f}")
print(f"Aleatoric Uncertainty: {uncertainties['aleatoric']:.4f}")
print(f"Epistemic Uncertainty: {uncertainties['epistemic']:.4f}")

# 4. Generate predictions from ensemble
print("\nGenerating from ensemble...")
outputs = model.generate_ensemble(prompt, max_length=100)

print("\nEnsemble predictions:")
for i, output in enumerate(outputs):
    print(f"  Member {i+1}: {output.split('[/INST]')[-1].strip()[:100]}")

# 5. Detect hallucination (requires trained detector)
print("\nTo detect hallucinations:")
print("1. Fine-tune the model on your dataset")
print("2. Collect uncertainty estimates and labels")
print("3. Train a detector:")
print("""
detector = HallucinationDetector(classifier_type='rf')
detector.train(uncertainties_list, labels)
is_hallucinated = detector.predict(uncertainties)
""")

print("\n✓ Quick start complete!")
print("See examples/faithfulness_detection.py for full training example")
