Data handoff

Version: 44e3694753a6   seed: 42

train: 4731 emails
val: 592 emails
test: 592 emails

Columns are email_id,text,label.  1 = spam, 0 = legit.
Use train to fit models, val to pick settings, and test only for final evaluation.
Keep these exact splits so the comparision is fair. Dont split them again.

The full files are in SmartMail_data_44e3694753a6.zip, kept seperate from Git.
The ZIP still needs to be shared through the group's shared storage.
Unzip it into data/processed/. It contains train.csv, val.csv, test.csv and manifest.json.
The manifest has a SHA256 for each CSV so everyone can check they have the same files.

ZIP SHA256: cabc52a3bb7453cb4fd20a637c6d3a9584d646e5d4cbd329919dd23227231a0e

The small sample in data/samples/synthetic_sample.csv is made up and only for trying loaders.
Do not add it to training data or use it for reported results.

Cleanup removed 131 exact repeats. All 581 detected near-duplicate pairs stay in one split.
See reports/data_summary.json for counts and the duplicate rule's limitations.
The Python scripts are still being reviewed locally.
