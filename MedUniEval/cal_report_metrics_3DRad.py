import json,os,sys
from utils.eval_3d import evaluate_3drad

sys.setrecursionlimit(400000)

def save_json(filename, ds):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(ds, f, indent=4, ensure_ascii=False)

print("\n" + "="*80)
print("Starting 3D-RAD Evaluation")
print("="*80)
output_path = "workspace/eval/Qwen3-VL/3DRad/"
# Method 1: synchronous loading
with open(output_path+"results.json", "r") as f:
    out_samples = json.load(f)

TYPE_MAPPING = {
    "Medical_Computation": "Medical Computation",
    "Spatial_Relationship": "Spatial Relationship",
    "Abnormality_Detection": "Abnormality Detection",
    "Organ_Identification": "Organ Identification",
    "Image_Quality": "Image Quality"
}

SUB_TYPE_MAPPING = {
    "Thickness": "Thickness",
    "Distance": "Distance",
    "Volume": "Volume",
    "Density": "Density",
    "Location": "Location",
    "Relative_Position": "Relative Position",
    "Presence": "Presence",
    "Severity": "Severity",
    "Organ_Name": "Organ Name",
    "Organ_Count": "Organ Count",
    "Contrast": "Contrast",
    "Noise": "Noise"
}

results_text, metrics, wrong_answers = evaluate_3drad(out_samples)

print(results_text)

wrong_answers_readable = {}
for key, value in wrong_answers.items():
    readable_key = TYPE_MAPPING.get(key, key)
    wrong_answers_readable[readable_key] = value

os.makedirs(output_path, exist_ok=True)
wrong_answers_path = os.path.join(output_path, "wrong_answers.json")

with open(wrong_answers_path, 'w', encoding='utf-8') as f:
    json.dump(wrong_answers_readable, f, indent=4, ensure_ascii=False)

print(f"\n✅ Wrong answers saved to: {wrong_answers_path}")
print("="*80 + "\n")

results_path = os.path.join(output_path, "results.json")
matric_path = os.path.join(output_path, "metrics.json")
save_json(matric_path, metrics)
save_json(results_path, out_samples)