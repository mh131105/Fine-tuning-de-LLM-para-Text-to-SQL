import os

def export_tables():
    os.makedirs("outputs/report_assets", exist_ok=True)
    
    # Spider
    with open("outputs/report_assets/spider_results_table.md", "w") as f:
        f.write("# Spider Results\n\n| Model | Execution Accuracy |\n|---|---|\n| Baseline | 0.0 |\n| Exp A | 0.0 |\n| Exp B | 0.0 |\n")
        
    # MMLU
    with open("outputs/report_assets/mmlu_regression_table.md", "w") as f:
        f.write("# MMLU Regression\n\n| Model | Accuracy |\n|---|---|\n| Baseline | 0.0 |\n| Exp A | 0.0 |\n| Exp B | 0.0 |\n")

if __name__ == "__main__":
    export_tables()
