#!/usr/bin/env python3
"""
📄 OMNI COMPILER
Compiles the 72 raw log files from the Omni Tuner into a single readable Markdown report.
"""

import os
import glob
import re

LOG_DIR = "artifacts/logs/omni_tune"
OUT_FILE = "artifacts/omni_tune_FULL_RESPONSES.md"

def extract_text(log_path):
    text = ""
    with open(log_path, "r") as f:
        for line in f:
            if "[DBG: Decoded" in line:
                try:
                    token = line.split("'")[1].replace("\\n", "\n")
                    text += token
                except:
                    pass
    return text.strip()

def parse_filename(filename):
    # run_231747_B0.8_R-1.0.txt
    parts = filename.replace(".txt", "").split("_")
    timestamp = parts[1]
    blend = parts[2].replace("B", "")
    repulsion = parts[3].replace("R", "")
    return timestamp, blend, repulsion

def main():
    print(f"Compiling logs from {LOG_DIR}...")
    
    log_files = glob.glob(os.path.join(LOG_DIR, "*.txt"))
    log_files.sort()
    
    with open(OUT_FILE, "w") as f_out:
        f_out.write("# 🌈 Omni-Tuner Results: Full Text Responses\n\n")
        f_out.write(f"**Total Runs:** {len(log_files)}\n\n")
        
        for log_file in log_files:
            try:
                fname = os.path.basename(log_file)
                ts, blend, rep = parse_filename(fname)
                text = extract_text(log_file)
                
                f_out.write(f"## 🧪 Run: Blend {blend} | Repulsion {rep}\n")
                f_out.write(f"*(Log: {fname})*\n\n")
                f_out.write("```\n")
                f_out.write(text)
                f_out.write("\n```\n")
                f_out.write("---\n\n")
            except Exception as e:
                print(f"Skipping {log_file}: {e}")
                
    print(f"✅ DONE. Compiled to {OUT_FILE}")

if __name__ == "__main__":
    main()
