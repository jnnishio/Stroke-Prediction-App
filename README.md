# Stroke Probability Prediction Application 
Built by Jason Nishio

>Stroke is a leading cause of death and serious long-term disability, causing about 1 in 6 cardiovascular-related deaths. **1 in 4 adults over age 25 will have a stroke in their lifetime**. Someone in the world has a stroke every 40 seconds, and someone dies of a stroke every 3 minutes and 14 seconds. Roughly 7 million adults are concurrently living with stroke in the US. 
>
>This app allows people of any age group who possess their basic health information to predict their likeliness of getting a stroke in the future. This can help people understand their need to seek further professional medical advice, receive necessary steps to prevent it, and intervene early. With a user-friendly app that may be widely distributed, more people can be prepared to encounter a stroke and be informed of when to receive care. Early identification and intervention of stroke has been statistically proven to reduce mortality rates.

## How to Use:
### Pre-requisites/Installations

* **Ensure Python is installed:**
    * Open **Terminal/Command Prompt**
    
        Mac: Open "Terminal" app
        
        Windows: press [Windows key], type `cmd` or `terminal`, press [Enter]
    * Type `python --version` and press enter. If a version name is not returned, install Python through [the website](python.org/downloads) or Microsoft Store. Once installed, check if it's installed or not again with `python --version`.
* **Install required Python packages:**
    * Open **Terminal/Command Prompt** (instructions above)
    * Run: `pip install gradio scikit-learn joblib numpy
`

### Usage Steps

1. Extract ZIP folder `Stroke_Prediction-Java_Final.ZIP`
2. Locate `app.py` in folder
3. **Copy its filepath**

    Mac: [**right click**] the file, hold [**option**], click **"Copy 'app.py' as Pathname"**
    
    Windows: hold [**Shift**], [**right click**] the file, select **"Copy as path"**

4. Open **Terminal/Command Prompt**
    
    Mac: Open "Terminal" app
    
    Windows: press [Windows key], type `cmd` or `terminal`, press [Enter]

5. Type `python [insert_file_path]`

6. Enter patient details in the interface automatically opened in your browser:
    * Gender
    * Age
    * Hypertension status
    * Heart Disease status
    * Residence type
    * Work type
    * Average gluocose level (may need to obtain information from medical professional)
    * BMI

7. Note any warnings outputted by the system!

8. (Optional): Click "**Flag**" to save your output as a csv! It should be saved at `~/.gradio/flagged/`

_Note: To exit the process in terminal, you may need to press (ctrl+C)._


## Machine Learning source code
Google Colab: https://colab.research.google.com/drive/191KesLHRO7UUZiiaeeKG5pM6ZlRp2iiK?usp=sharing