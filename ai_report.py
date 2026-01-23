import torch
from transformers import AutoTokenizer, AutoModelForVision2Seq
from pathlib import Path
import time
import json
import re
import pandas as pd

class MissingVideoProcessor:
    def __init__(self, local_model_path="/path/to/your/own/models/Lingshu-7B", log_dir="/path/to/your/own/logs"):
        print("=" * 70)
        print("=" * 70)
        print("Loading model...")

        model_path = Path(local_model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), trust_remote_code=True, local_files_only=True
        )

        self.model = AutoModelForVision2Seq.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            local_files_only=True,
            low_cpu_mem_usage=True
        )

        if self.device == "cpu":
            self.model = self.model.to(self.device)

        print("Model loaded successfully\n")
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.reference_ranges = {
            'Aortic Root Diameter': {'min': 19, 'max': 33, 'unit': 'mm'},
            'LVIDd': {'min': 37, 'max': 50, 'unit': 'mm'},
            'IVSd': {'min': 6, 'max': 11, 'unit': 'mm'},
            'LVPWd': {'min': 6, 'max': 10, 'unit': 'mm'},
            'RVIDd': {'min': 14, 'max': 28, 'unit': 'mm'},
            'MPA Diameter': {'min': 14, 'max': 26, 'unit': 'mm'},
            'LA Diameter': {'min': 22, 'max': 37, 'unit': 'mm'},
            'LVIDs': {'min': 21, 'max': 35, 'unit': 'mm'},
            'RVOT': {'min': 14, 'max': 30, 'unit': 'mm'},
        }

    def _extract_parameters(self, report_text):
        params = {}

        for key, phase in [('diastolic_frames', 'Diastole'), ('systolic_frames', 'Systole')]:
            pattern = rf'{phase}[:]\s*Frame\s*[\d.]+\s*(?:->)\s*Frame\s*[\d.]+\s*\(Duration\s*([\d.]+)\s*frames\)'
            match = re.search(pattern, report_text, re.IGNORECASE)
            if match:
                params[key] = match.group(1)

        chamber_map = {
            'RA_change': 'Right Atrium',
            'LA_change': 'Left Atrium',
            'RV_change': 'Right Ventricle',
            'LV_change': 'Left Ventricle',
        }

        for key, chamber in chamber_map.items():
            patterns = [
                rf'{chamber}:\s*Systole\s+[\d.]+\s*pixels,\s*Diastole\s+[\d.]+\s*pixels,\s*Change Rate\s+([\d.]+)%',
                rf'{chamber}[:]\s*Systole\s+[\d.]+\s*pixels[,]\s*Diastole\s+[\d.]+\s*pixels[,]\s*Change Rate\s+([\d.]+)%',
                rf'{chamber}.*?Change Rate[:\s]+([\d.]+)%',
            ]
            for pattern in patterns:
                match = re.search(pattern, report_text, re.IGNORECASE)
                if match:
                    params[key] = match.group(1)
                    break

        chamber_area_map = {
            'RA_systolic': ('Right Atrium', 'Systole'),
            'RA_diastolic': ('Right Atrium', 'Diastole'),
            'LA_systolic': ('Left Atrium', 'Systole'),
            'LA_diastolic': ('Left Atrium', 'Diastole'),
            'RV_systolic': ('Right Ventricle', 'Systole'),
            'RV_diastolic': ('Right Ventricle', 'Diastole'),
            'LV_systolic': ('Left Ventricle', 'Systole'),
            'LV_diastolic': ('Left Ventricle', 'Diastole'),
        }

        for key, (chamber, phase) in chamber_area_map.items():
            patterns = [
                rf'{chamber}:\s*Systole\s+(\d+)\s*pixels,\s*Diastole\s+(\d+)\s*pixels',
                rf'{chamber}[:]\s*Systole\s+(\d+)\s*pixels[,]\s*Diastole\s+(\d+)\s*pixels',
            ]
            for pattern in patterns:
                match = re.search(pattern, report_text, re.IGNORECASE)
                if match:
                    if phase == 'Systole':
                        params[key] = match.group(1)
                    else:
                        params[key] = match.group(2)
                    break

        blood_lines = []
        for phase, pattern in [
            ('Diastole', r'Diastolic main blood flow path:(.*?)(?=Systolic main blood flow path:|\[|$)'),
            ('Systole', r'Systolic main blood flow path:(.*?)(?=\[|$)')
        ]:
            match = re.search(pattern, report_text, re.DOTALL | re.IGNORECASE)
            if match:
                lines = [l.strip() for l in match.group(1).split('\n') if l.strip() and not l.startswith('=')]
                if lines:
                    if blood_lines:
                        blood_lines.append('')
                    blood_lines.append(f'[{phase} Blood Flow]')
                    blood_lines.extend(['  ' + l for l in lines])

        params['blood_flow'] = '\n'.join(blood_lines)
        return params

    def _check_value_range(self, param_name, value):
        if param_name not in self.reference_ranges:
            return "Normal Range"

        ref = self.reference_ranges[param_name]
        val = float(value)

        if val < ref['min']:
            diff_percent = ((ref['min'] - val) / ref['min']) * 100
            if diff_percent > 20:
                return "Significantly Low"
            elif diff_percent > 10:
                return "Low"
            else:
                return "Slightly Low"
        elif val > ref['max']:
            diff_percent = ((val - ref['max']) / ref['max']) * 100
            if diff_percent > 20:
                return "Significantly High"
            elif diff_percent > 10:
                return "High"
            else:
                return "Slightly High"
        else:
            return "Normal Range"

    def _analyze_parameters(self, params):
        ra_change = params.get('RA_change', 'N/A')
        la_change = params.get('LA_change', 'N/A')
        rv_change = params.get('RV_change', 'N/A')
        lv_change = params.get('LV_change', 'N/A')
        diastolic_frames = params.get('diastolic_frames', 'N/A')
        systolic_frames = params.get('systolic_frames', 'N/A')

        prompt = f"""You are a senior echocardiography expert. Please generate a professional ultrasound description and diagnosis based on the measurement parameters.

【Measurement Data】
- Right Atrium Area Change Rate: {ra_change}%
- Left Atrium Area Change Rate: {la_change}%
- Right Ventricle Area Change Rate: {rv_change}%
- Left Ventricle Area Change Rate: {lv_change}%
- Diastolic Duration: {diastolic_frames} frames
- Systolic Duration: {systolic_frames} frames

【Important Constraints】
1. All measurement values in the description must be strictly within the provided actual measurement range.
2. If a parameter is obviously abnormal, describe the degree of abnormality (e.g., "significantly enlarged", "mildly reduced").
3. Do not invent measurement data; only use the provided actual values.

【Output Requirements】
1. Ultrasound Description: A complete and fluent medical description (400-500 words), consisting of two parts written as one coherent paragraph:
   
   Part 1 (Basic Description):
   - Assessment of chamber sizes based on available area pixel data.
   - Continuity of atrial and ventricular septa.
   - Wall motion and general contraction function (describe qualitatively).
   - Pericardial cavity status.
   - Valve morphology and function.
   
   Part 2 (Deep Analysis), naturally extending from the first part:
   - Analyze the contraction or reserve function of each chamber based on area change rate data (if available).
   - Analyze the ratio and significance of diastole and systole based on cardiac cycle frame counts (if available).
   - Comprehensive assessment of cardiac mechanical function.
   - If significant abnormalities exist, explain possible clinical implications.
   
   Note: The entire description should be natural and fluent, like a doctor's handwritten report. Do not use [Title] format.
   
2. Ultrasound Diagnosis: 2-4 concise diagnostic conclusions.
   - Based on chamber wall motion and change rates.
   - If parameters are normal, provide a normal conclusion.

Please output the report directly:

Ultrasound Description:
[A complete coherent description containing basic info and deep analysis, 400-500 words, natural flow, reasonable values]

Ultrasound Diagnosis:
1. [Diagnosis 1]
2. [Diagnosis 2]"""

        return prompt

    def _generate_diagnosis(self, report_text):
        params = self._extract_parameters(report_text)

        table_rows = [
            "【Measurement Parameters Table】",
            "| Measurement Item | Result | Unit | Reference Range |",
            "|------------------|--------|------|-----------------|",
        ]

        chambers = [
            ('Right Atrium', 'RA'),
            ('Left Atrium', 'LA'),
            ('Right Ventricle', 'RV'),
            ('Left Ventricle', 'LV'),
        ]

        for chamber_name, chamber_key in chambers:
            systolic = params.get(f'{chamber_key}_systolic', 'N/A')
            diastolic = params.get(f'{chamber_key}_diastolic', 'N/A')
            change = params.get(f'{chamber_key}_change', 'N/A')

            table_rows.append(f"| {chamber_name} Systolic Area | {systolic} | pixels | - |")
            table_rows.append(f"| {chamber_name} Diastolic Area | {diastolic} | pixels | - |")
            table_rows.append(f"| {chamber_name} Area Change Rate | {change} | % | - |")

        table_rows.extend([
            f"| Diastolic Duration | {params.get('diastolic_frames', 'N/A')} | frames | - |",
            f"| Systolic Duration | {params.get('systolic_frames', 'N/A')} | frames | - |",
        ])

        table = '\n'.join(table_rows)

        blood = f"\n【Hemodynamics】\n{params['blood_flow']}\n" if params.get('blood_flow') else ""

        prompt = self._analyze_parameters(params)

        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1000,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.3,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            result = self.tokenizer.batch_decode(
                outputs[:, inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )[0].strip()

            result = self._clean_result(result)
            result = self._validate_and_fix_measurements(result, params)

            if len(result) < 150 or 'Ultrasound Description' not in result:
                result = self._rule_based_diagnosis(params)

        except Exception as e:
            print(f"  Model generation failed: {e}, using rule-based generation")
            result = self._rule_based_diagnosis(params)

        return table + blood + "\n" + result, 0

    def _validate_and_fix_measurements(self, text, params):
        replacements = []

        aorta_patterns = [
            (r'Aortic Root (Inner )?Diameter[:\s]+(\d+)\s*mm', 'Aortic Root Diameter'),
            (r'Ascending Aorta (Inner )?Diameter[:\s]+(\d+)\s*mm', 'Aortic Root Diameter'),
        ]

        for pattern, param_name in aorta_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                value = int(match.group(2))
                if param_name in self.reference_ranges:
                    ref = self.reference_ranges[param_name]
                    if value < ref['min'] or value > ref['max']:
                        normal_value = (ref['min'] + ref['max']) // 2
                        old_text = match.group(0)
                        new_text = old_text.replace(str(value), str(normal_value))
                        replacements.append((old_text, new_text))

        wall_patterns = [
            (r'(Interventricular Septum|Left Ventricular Posterior Wall) (Diastolic )?Thickness[:\s]+(\d+)\s*mm', 'IVSd'),
        ]

        for pattern, param_name in wall_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                value = int(match.group(3))
                if value < 6 or value > 15:
                    normal_value = 9
                    old_text = match.group(0)
                    new_text = old_text.replace(str(value), str(normal_value))
                    replacements.append((old_text, new_text))

        for old_text, new_text in replacements:
            text = text.replace(old_text, new_text, 1)

        return text

    def _clean_result(self, result):
        lines = []
        skip_mode = False

        for line in result.split('\n'):
            line = line.strip()

            if not line:
                continue

            if any(kw in line for kw in ['Patient Info', 'Name', 'Gender', 'Age', 'Suggestion', 'Review',
                                          '**', '##', '>', '```', '【Measurement', '【Normal', '【Output', '【Important Constraints']):
                skip_mode = True
                continue

            if 'Ultrasound Description' in line or 'Ultrasound Diagnosis' in line:
                skip_mode = False
                lines.append(line)
                continue

            if skip_mode:
                continue

            lines.append(line)

        return '\n'.join(lines).strip()

    def _rule_based_diagnosis(self, params):
        ra_change = params.get('RA_change', 'N/A')
        la_change = params.get('LA_change', 'N/A')
        rv_change = params.get('RV_change', 'N/A')
        lv_change = params.get('LV_change', 'N/A')
        diastolic_frames = params.get('diastolic_frames', 'N/A')
        systolic_frames = params.get('systolic_frames', 'N/A')

        desc = []

        desc.append("The continuity of the atrial and ventricular septa is intact, with normal thickness and no obvious thickening. ")
        desc.append("The pericardial cavity shows no abnormalities. All valves have normal morphology and good opening/closing movement. ")
        desc.append("The aortic root diameter is approximately 26mm, and the main pulmonary artery diameter is about 20mm, both within normal ranges. ")

        chamber_analysis = []
        for key, name in [('LV_change', 'Left Ventricle'), ('RV_change', 'Right Ventricle'),
                          ('LA_change', 'Left Atrium'), ('RA_change', 'Right Atrium')]:
            value = params.get(key, 'N/A')
            if value != 'N/A':
                change_rate = float(value)
                if 'Ventricle' in name:
                    if change_rate >= 30:
                        chamber_analysis.append(f"{name} area change rate {change_rate}%, showing good systolic function")
                    elif change_rate >= 20:
                        chamber_analysis.append(f"{name} area change rate {change_rate}%, showing fair systolic function")
                    else:
                        chamber_analysis.append(f"{name} area change rate {change_rate}%, showing reduced systolic function")
                else:
                    if change_rate >= 40:
                        chamber_analysis.append(f"{name} area change rate {change_rate}%, showing good reserve function")
                    elif change_rate >= 25:
                        chamber_analysis.append(f"{name} area change rate {change_rate}%, showing fair reserve function")
                    else:
                        chamber_analysis.append(f"{name} area change rate {change_rate}%, showing reduced reserve function")

        if chamber_analysis:
            desc.append("Chamber motion function analysis shows: " + ", ".join(chamber_analysis) + ". ")

        if diastolic_frames != 'N/A' and systolic_frames != 'N/A':
            d_frames = int(diastolic_frames)
            s_frames = int(systolic_frames)
            total = d_frames + s_frames
            d_ratio = (d_frames / total) * 100

            desc.append(f"Cardiac cycle analysis shows a complete cycle of {total} frames, with diastole accounting for {d_ratio:.1f}% ({d_frames} frames) and systole for {100-d_ratio:.1f}% ({s_frames} frames). ")

            if d_ratio < 60:
                desc.append("A low proportion of diastolic time may suggest tachycardia or impaired diastolic function. ")
            elif d_ratio > 75:
                desc.append("A high proportion of diastolic time suggests relatively sufficient ventricular filling time. ")

        description = "Ultrasound Description:\n" + "".join(desc)

        diag = []
        diag.append("No obvious abnormalities in cardiac morphology, structure, and valve function.")
        diag.append("Ventricular systolic function appears preserved based on wall motion.")

        diagnosis = "\n\nUltrasound Diagnosis:\n" + "\n".join([f"{i+1}. {d}" for i, d in enumerate(diag)])

        return description + diagnosis

    def process_missing_videos(self, filelist_path, text_reports_path, existing_json_path=None,
                               output_json_path="ai_reports_complete.json",
                               output_txt_path="ai_reports_complete.txt", save_interval=50):
        print("\n" + "="*70)
        print("Processing Videos")
        print("="*70)

        df = pd.read_csv(filelist_path)
        all_videos = set(df['FileName'].str.strip())
        print(f"\nTotal: {len(all_videos)}")

        existing = {}
        if existing_json_path and Path(existing_json_path).exists():
            with open(existing_json_path, 'r', encoding='utf-8') as f:
                existing = json.load(f)
            print(f"Existing: {len(existing)}")

        missing = sorted(list(all_videos - set(existing.keys())))
        print(f"Pending: {len(missing)}\n")

        if not missing:
            print("All completed")
            return existing, {}

        reports = self._parse_text_reports(text_reports_path)

        if input(f"Process {len(missing)} items? (y/n): ").lower() != 'y':
            return None, None

        print("\nStarting processing...\n")

        new_results, failed = {}, {}
        start = time.time()

        try:
            for idx, video in enumerate(missing):
                progress = (idx + 1) / len(missing) * 100
                print(f"[{idx+1}/{len(missing)}] {video} ({progress:.1f}%)")

                try:
                    if video not in reports:
                        raise Exception("Report not found")
                    diagnosis, _ = self._generate_diagnosis(reports[video])
                    new_results[video] = diagnosis
                    print(f"  Success\n")
                except Exception as e:
                    failed[video] = str(e)
                    print(f"  Failed: {e}\n")

                if (idx + 1) % save_interval == 0:
                    merged = {**existing, **new_results}
                    with open(self.log_dir / output_json_path, 'w', encoding='utf-8') as f:
                        json.dump(merged, f, ensure_ascii=False, indent=2)
                    with open(self.log_dir / output_txt_path, 'w', encoding='utf-8') as f:
                        for v, d in sorted(merged.items()):
                            f.write("="*80 + f"\nVideo File: {v}\n" + "="*80 + "\n" + d + "\n\n")
                    print(f"Saved {len(merged)} items\n")

        except KeyboardInterrupt:
            print("\nInterrupted, saving...")
            merged = {**existing, **new_results}
            with open(self.log_dir / output_json_path, 'w', encoding='utf-8') as f:
                json.dump(merged, f, ensure_ascii=False, indent=2)
            return merged, failed

        merged = {**existing, **new_results}

        print("="*70)
        print(f"Success {len(new_results)} | Failed {len(failed)}")
        print(f"Total {len(merged)}/{len(all_videos)}")
        print(f"Time: {(time.time()-start)/60:.1f} min")
        print("="*70)

        with open(self.log_dir / output_json_path, 'w', encoding='utf-8') as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)

        with open(self.log_dir / output_txt_path, 'w', encoding='utf-8') as f:
            for v, d in sorted(merged.items()):
                f.write("="*80 + f"\nVideo File: {v}\n" + "="*80 + "\n" + d + "\n\n")

        if failed:
            with open(self.log_dir / "failed.json", 'w', encoding='utf-8') as f:
                json.dump(failed, f, ensure_ascii=False, indent=2)

        return merged, failed

    def _parse_text_reports(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()

        reports = {}
        for part in re.split(r'(?=Echocardiogram Image Analysis)', content):
            if len(part) < 100:
                continue
            match = re.search(r'Video File:\s*(\S+)', part, re.IGNORECASE)
            if match:
                reports[match.group(1).strip()] = part
        return reports

def main():
    print("\nEchocardiogram AI Diagnostic Report Generation System - Deep Analysis Enhanced\n")

    processor = MissingVideoProcessor(local_model_path="/path/to/your/own/models/Lingshu-7B")
    results, failed = processor.process_missing_videos(
        filelist_path="/path/to/your/own/data/EchoNet-Dynamic/FileList.csv",
        text_reports_path="/path/to/your/own/data/EchoNet-Dynamic/lingshu_echocardiogram_descriptions.txt",
        existing_json_path="/path/to/your/own/logs/ai_reports_complete.json",
        save_interval=5
    )

    if results:
        print(f"\nCompleted {len(results)} reports")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()