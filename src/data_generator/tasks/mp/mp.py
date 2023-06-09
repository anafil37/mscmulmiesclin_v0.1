import os
import re
import sys

sys.path.append('.')

import pandas as pd


import os
import csv


def filter_notes(notes_df: pd.DataFrame, admissions_df: pd.DataFrame, admission_text_only=False) -> pd.DataFrame:
    """
    Keep only Discharge Summaries and filter out Newborn admissions. Replace duplicates and join reports with
    their addendums. If admission_text_only is True, filter all sections that are not known at admission time.
    """
    # filter out newborns
    adm_grownups = admissions_df[admissions_df.ADMISSION_TYPE != "NEWBORN"]
    notes_df = notes_df[notes_df.HADM_ID.isin(adm_grownups.HADM_ID)]

    # remove notes with no TEXT or HADM_ID
    notes_df = notes_df.dropna(subset=["TEXT", "HADM_ID"])

    # filter discharge summaries
    notes_df = notes_df[notes_df.CATEGORY == "Discharge summary"]

    # remove duplicates and keep the later ones
    notes_df = notes_df.sort_values(by=["CHARTDATE"])
    notes_df = notes_df.drop_duplicates(subset=["TEXT"], keep="last")

    # combine text of same admissions (those are usually addendums)
    combined_adm_texts = notes_df.groupby('HADM_ID')['TEXT'].apply(lambda x: '\n\n'.join(x)).reset_index()
    notes_df = notes_df[notes_df.DESCRIPTION == "Report"]
    notes_df = notes_df[["HADM_ID", "ROW_ID", "SUBJECT_ID", "CHARTDATE"]]
    notes_df = notes_df.drop_duplicates(subset=["HADM_ID"], keep="last")
    notes_df = pd.merge(combined_adm_texts, notes_df, on="HADM_ID", how="inner")

    # strip texts from leading and trailing and white spaces
    notes_df["TEXT"] = notes_df["TEXT"].str.strip()

    # remove entries without admission id, subject id or text
    notes_df = notes_df.dropna(subset=["HADM_ID", "SUBJECT_ID", "TEXT"])

    if admission_text_only:
        # reduce text to admission-only text
        notes_df = filter_admission_text(notes_df)

    return notes_df


def filter_admission_text(notes_df) -> pd.DataFrame:
    """
    Filter text information by section and only keep sections that are known on admission time.
    """
    admission_sections = {
        "CHIEF_COMPLAINT": "chief complaint:",
        "PRESENT_ILLNESS": "present illness:",
        "MEDICAL_HISTORY": "medical history:",
        "MEDICATION_ADM": "medications on admission:",
        "ALLERGIES": "allergies:",
        "PHYSICAL_EXAM": "physical exam:",
        "FAMILY_HISTORY": "family history:",
        "SOCIAL_HISTORY": "social history:"
    }

    # replace linebreak indicators
    notes_df['TEXT'] = notes_df['TEXT'].str.replace(r"\n", r"\\n")

    # extract each section by regex
    for key in admission_sections.keys():
        section = admission_sections[key]
        notes_df[key] = notes_df.TEXT.str.extract(r'(?i){}(.+?)\\n\\n[^(\\|\d|\.)]+?:'
                                                  .format(section))

        notes_df[key] = notes_df[key].str.replace(r'\\n', r' ')
        notes_df[key] = notes_df[key].str.strip()
        notes_df[key] = notes_df[key].fillna("")
        notes_df[notes_df[key].str.startswith("[]")][key] = ""

    # filter notes with missing main information
    notes_df = notes_df[(notes_df.CHIEF_COMPLAINT != "") | (notes_df.PRESENT_ILLNESS != "") |
                        (notes_df.MEDICAL_HISTORY != "")]

    # add section headers and combine into TEXT_ADMISSION
    notes_df = notes_df.assign(TEXT="CHIEF COMPLAINT: " + notes_df.CHIEF_COMPLAINT.astype(str)
                                    + '\n\n' +
                                    "PRESENT ILLNESS: " + notes_df.PRESENT_ILLNESS.astype(str)
                                    + '\n\n' +
                                    "MEDICAL HISTORY: " + notes_df.MEDICAL_HISTORY.astype(str)
                                    + '\n\n' +
                                    "MEDICATION ON ADMISSION: " + notes_df.MEDICATION_ADM.astype(str)
                                    + '\n\n' +
                                    "ALLERGIES: " + notes_df.ALLERGIES.astype(str)
                                    + '\n\n' +
                                    "PHYSICAL EXAM: " + notes_df.PHYSICAL_EXAM.astype(str)
                                    + '\n\n' +
                                    "FAMILY HISTORY: " + notes_df.FAMILY_HISTORY.astype(str)
                                    + '\n\n' +
                                    "SOCIAL HISTORY: " + notes_df.SOCIAL_HISTORY.astype(str))

    return notes_df


def save_mimic_split_patient_wise(df, label_column1, label_column2, save_dir, task_name, seed, column_list=None):
    """
    Splits a MIMIC dataframe into 70/10/20 train, val, test with no patient occuring in more than one set.
    Uses ROW_ID as ID column and save to save_path.
    """
    if column_list is None:
        column_list = ["ID", "TEXT", label_column1, label_column2]

    # Load prebuilt MIMIC patient splits
    data_split = {"train": pd.read_csv("/home/ana.lopes/mscmulmiesclin_v0.1/src/data_generator/tasks/mimic_train.csv"),
                  "val": pd.read_csv("/home/ana.lopes/mscmulmiesclin_v0.1/src/data_generator/tasks/mimic_val.csv"),
                  "test": pd.read_csv("/home/ana.lopes/mscmulmiesclin_v0.1/src/data_generator/tasks/mimic_test.csv")}

    # Use row id as general id and cast to int
    df = df.rename(columns={'HADM_ID': 'ID'})
    df.ID = df.ID.astype(int)

    # Create path to task data
    os.makedirs(save_dir, exist_ok=True)

    # Save splits to data folder
    for split_name in ["train", "val", "test"]:
        split_set = df[df.SUBJECT_ID.isin(data_split[split_name].SUBJECT_ID)].sample(frac=1,
                                                                                     random_state=seed)[column_list]

        # lower case column names
        split_set.columns = map(str.lower, split_set.columns)

        split_set.to_csv(os.path.join(save_dir, "{}_{}.csv".format(task_name, split_name)),
                         index=False,
                         quoting=csv.QUOTE_ALL)
        
def mp_in_hospital_mimic(mimic_dir: str, save_dir: str, seed: int, admission_only: bool):
    """
    Extracts information needed for the task from the MIMIC dataset. Namely "TEXT" column from NOTEEVENTS.csv and
    "HOSPITAL_EXPIRE_FLAG" from ADMISSIONS.csv. Filters specific admission sections for often occuring signal words.
    Creates 70/10/20 split over patients for train/val/test sets.
    """

    # set task name
    task_name = "MP_IN"
    if admission_only:
        task_name = f"{task_name}_adm"

    # load dataframes
    mimic_diagnoses = pd.read_csv(os.path.join(mimic_dir, "DIAGNOSES_ICD.csv"))
    mimic_notes = pd.read_csv(os.path.join(mimic_dir, "NOTEEVENTS.csv"))
    mimic_admissions = pd.read_csv(os.path.join(mimic_dir, "ADMISSIONS.csv"))

    # filter notes
    mimic_notes = filter_notes(mimic_notes, mimic_admissions, admission_text_only=admission_only)

    # append HOSPITAL_EXPIRE_FLAG to notes
    notes_with_expire_flag = pd.merge(mimic_notes, mimic_admissions[["HADM_ID", "HOSPITAL_EXPIRE_FLAG"]], how="left",
                                      on="HADM_ID")

    # drop all rows without hospital expire flag
    notes_with_expire_flag = notes_with_expire_flag.dropna(how='any', subset=['HOSPITAL_EXPIRE_FLAG'], axis=0)

    # filter out written out death indications
    notes_with_expire_flag = remove_mentions_of_patients_death(notes_with_expire_flag)

    # only keep relevant columns
    mimic_diagnoses = mimic_diagnoses[['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE']]

    # drop all rows without diagnosis codes
    mimic_diagnoses = mimic_diagnoses.dropna(
        how='any', subset=['ICD9_CODE'], axis=0)

    # create column SHORT_CODE including first 3 digits of ICD9 code
    mimic_diagnoses["SHORT_CODE"] = mimic_diagnoses.ICD9_CODE.astype(str)

    mimic_diagnoses.loc[
        mimic_diagnoses['SHORT_CODE'].str.startswith("V"), 'SHORT_CODE'] = mimic_diagnoses.SHORT_CODE.apply(
        lambda x: x[:4])
    mimic_diagnoses.loc[
        mimic_diagnoses['SHORT_CODE'].str.startswith("E"), 'SHORT_CODE'] = mimic_diagnoses.SHORT_CODE.apply(
        lambda x: x[:4])
    mimic_diagnoses.loc[(~mimic_diagnoses.SHORT_CODE.str.startswith("E")) & (
        ~mimic_diagnoses.SHORT_CODE.str.startswith("V")), 'SHORT_CODE'] = mimic_diagnoses.SHORT_CODE.apply(
        lambda x: x[:3])

    # remove duplicated code groups per admission
    mimic_diagnoses = mimic_diagnoses.drop_duplicates(
        ["HADM_ID", "SHORT_CODE"])

    # store all ICD codes for vectorization
    icd9_codes = mimic_diagnoses.SHORT_CODE.unique().tolist()

    grouped_codes = mimic_diagnoses.groupby(['HADM_ID', 'SUBJECT_ID'])['SHORT_CODE'].apply(
        lambda d: ",".join(d.astype(str))).reset_index()

    # rename column
    grouped_codes = grouped_codes.rename(columns={'SHORT_CODE': 'SHORT_CODES'})

    # append ICD9 codes to notes
    notes_with_expire_flag_and_icd9 = pd.merge( notes_with_expire_flag, 
        grouped_codes[['HADM_ID', 'SHORT_CODES']], how='inner', on='HADM_ID')

    save_mimic_split_patient_wise(notes_with_expire_flag_and_icd9,
                                              label_column1='HOSPITAL_EXPIRE_FLAG',
                                              label_column2='SHORT_CODES',
                                              save_dir=save_dir,
                                              task_name=task_name,
                                              seed=seed)
    



def remove_mentions_of_patients_death(df: pd.DataFrame):
    """
    Some notes contain mentions of the patient's death such as 'patient deceased'. If these occur in the sections
    PHYSICAL EXAM and MEDICATION ON ADMISSION, we can simply remove the mentions, because the conditions are not
    further elaborated in these sections. However, if the mentions occur in any other section, such as CHIEF COMPLAINT,
    we want to remove the whole sample, because the patient's passing if usually closer described in the text and an
    outcome prediction does not make sense in these cases.
    """

    death_indication_in_special_sections = re.compile(
        r"((?:PHYSICAL EXAM|MEDICATION ON ADMISSION):[^\n\n]*?)((?:patient|pt)?\s+(?:had\s|has\s)?(?:expired|died|passed away|deceased))",
        flags=re.IGNORECASE)

    death_indication_in_all_other_sections = re.compile(
        r"(?:patient|pt)\s+(?:had\s|has\s)?(?:expired|died|passed away|deceased)", flags=re.IGNORECASE)

    # first remove mentions in sections PHYSICAL EXAM and MEDICATION ON ADMISSION
    df['TEXT'] = df['TEXT'].replace(death_indication_in_special_sections, r"\1", regex=True)

    # if mentions can be found in any other section, remove whole sample
    df = df[~df['TEXT'].str.contains(death_indication_in_all_other_sections)]

    # remove other samples with obvious death indications
    df = df[~df['TEXT'].str.contains("he expired", flags=re.IGNORECASE)]  # does also match 'she expired'
    df = df[~df['TEXT'].str.contains("pronounced expired", flags=re.IGNORECASE)]
    df = df[~df['TEXT'].str.contains("time of death", flags=re.IGNORECASE)]

    return df


if __name__ == "__main__":
    mimic_dir = "/home/ana.lopes/mscmulmiesclin_v0.1/data"
    save_dir = "/home/ana.lopes/mscmulmiesclin_v0.1/data"
    seed = 123
    admission_only = True
    mp_in_hospital_mimic(mimic_dir, save_dir, seed, admission_only)
