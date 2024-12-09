```bash
unrar x lau_dataset.rar
```

### Step 2: Clean the Data
To clean the dataset, similar to [OSRT](https://github.com/Fanghua-Yu/OSRT/blob/master/odisr/utils/make_clean_lau_dataset.py), run the following command:

```bash
python Data_prepare/make_clean_lau_dataset.py
```

This will create a `lau_dataset_clean` folder inside the `BPOSR/experiments` directory.

---