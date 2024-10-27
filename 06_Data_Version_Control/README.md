## EMLOV4-Session-06 Assignment - Data Version Control

### Contents

**Note: I have completed the optional assignment of integrating comet-ml**

- [Requirements](#requirements)
- [Development Method](#development-method)
    - [DVC Integration with Google Cloud Storage](#dvc-integration-with-google-cloud-storage)
    - [Integrate Comet ML](#integrate-comet-ml)
    - [Github Actions with DVC Pipeline for training](#github-actions-with-dvc-pipeline-for-training)
    - [Train-Test-Infer-Comment-CML](#train-test-infer-comment-cml)
- [Learnings](#learnings)
- [Results Screenshots](#results)

### Requirements

- Start with your repository from last session
- Add this dataset: https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip.
- Add DVC Integration with Google Drive
- Integrate CometML for logging
- Create a Github Actions with DVC Pipeline for training
- Train any ViT model for 5 epochs
- Here are the Plots you will show
    - train/acc and val/acc in one plot
    - train/loss and val/loss in one plot
    - Confusion Matrix for test dataset and train dataset as image plot
- Infer on 10 images from test dataset and display the prediction, target along with image in results.md.
    - You’ll be using your infer.py. script for this
    - You can save the images in the predictions folder and then add them to the results.md.
- Change Model to pretrained and create a PR

**Optional Assignment**

- Integrate `CometML` for logging

### Development Method

#### Build Command

**Debug Commands for development**

```docker build -t light_train_test -f ./Dockerfile .```

```docker run -d -v /workspace/emlo4-session-06-ajithvcoder/:/workspace/ light_train_test```

```docker exec -it <c511d4e6ed1a9ca6933c67f02632a2> /bin/bash```

**Train Test Infer Commands**

Install

```uv sync --extra-index-url https://download.pytorch.org/whl/cpu ```

Pull data from cloud

```dvc pull -r myremote1```

Trigger workflow

```dvc repro```

Comment in PR or commit
```cml comment create report.md```

### DVC Integration with Google Cloud Storage

- Follow first point in the `Using service account method `metioned here https://dvc.org/doc/user-guide/data-management/remote-storage/google-drive#using-service-accounts
- Store the api key in local folder as `credentials.json` but dont commit it to github. if u do so github will raise a warning but inturn google notifies it
and revokes the credentials.
- Better to give `owner` permission/`storage admin` permission to the user account 
- Create a folder in google bucket service and get the url for example - `gs://dvcmanager/storage` where `dvcmanager` is bucket name and `storage` is folder name
- After structuring the train and test images in data folder
- Run ```dvc init```
- Now run `dvc remote add -d myremote gs://<mybucket>/<path>` command. Reference https://dvc.org/doc/user-guide/data-management/remote-storage/google-cloud-storage
- Run ```dvc add data```
- Run ```dvc push -r myremote1```
- Wait for 10 minutes as its 800 MB and if its in github actions wait for 15 minutes.
- Now add data.yml each and every step using ```dvc stage add``` command

**Add Train, test, infer, report_generation stages**

- `dvc stage add -f -n train -d configs/experiment/catdog_ex.yaml -d src/train.py -d data/cat_dog_medium python src/train.py --config-name=train experiment=catdog_ex trainer.max_epochs=5`

- `dvc stage add -f -n test -d configs/experiment/catdog_ex_eval.yaml -d src/eval.py  python src/eval.py --config-name=eval experiment=catdog_ex_eval.yaml `

- `dvc stage add -f -n infer -d configs/experiment/catdog_ex_eval.yaml -d src/infer.py python src/infer.py --config-name=infer experiment=catdog_ex_eval.yaml` 

- `dvc stage add -n report_genration python scripts/metrics_fetch.py`

- You would have generated a `dvc.yaml` file, `data.dvc` file and `dvc.lock` file push all these to github


### Integrate Comet ML

- Comet-ML is already inegrated with pytorch lighting so we just need to add config files in "logger" folder and use proper api key for it.



### Github Actions with DVC Pipeline for training

- setup cml, uv packages using github actions and install `python=3.12`
- Copy the contents of credentials.json and store in github reprository secrets with name `GDRIVE_CREDENTIALS_DATA`

### Train-Test-Infer-Comment-CML

**Debugging and development**

Use a subset of train and test set for faster debugging and development. Also u can reduce the configs of model to generate a `custom 3 million param vit model`. I have reduced from 5 million params to 3 million params by using the config. However to run the pretrained model we can change this config.

**Overall Run**
- `dvc repro`

**Train**
- `dvc repro train`

**Test**
- `dvc repro test`

**Infer**
- `dvc repro infer`

**Create CML report**

- Install cml pacakge
- `python scripts/metrics_fetch.py` will fetch the necessary files needed for report and place it in root folder
- `report_gen.sh`collects and appends every metric to readme file
- cml tool is used to comment in github and it internally uses github token to authorize


### CI Pipeline Development

- Using GitHub Actions and the `dvc-pipeline.yml`, we are running all above actions and it could be triggered both manually and on pull request given to main branch


### Learnings

- Learnt about DVC tool usage, Comet ml, and cml

### Results

**Comet-ML Dashboard**

![comet ml dashboard](./assets/snap_comet_ml.png)

**Work flow success on main branch**

Run details - [here](https://github.com/ajithvcoder/emlo4-session-06-ajithvcoder/actions/runs/11419499613)

![main workflow](./assets/snap_main_workflow.png)

**Work flow success run on PR branch**

Run details - [here](https://github.com/ajithvcoder/emlo4-session-06-ajithvcoder/actions/runs/11419924207)

Pull request - [here](https://github.com/ajithvcoder/emlo4-session-06-ajithvcoder/pull/2)

![pr triggered workflow](./assets/snap_pr_testing.png)

**Comments from cml with plots and 10 infer images**

Details - [here](https://github.com/ajithvcoder/emlo4-session-06-ajithvcoder/pull/2#issuecomment-2424194445)

![cml comment](./assets/snap_cml.png)


Note: I used Google cloud Storage bucket for this project as it was faster than gdrive and its paid one so after successfully completing this assignment i am going to remove it. So you need to do the cloud setup again for re-running this experiment.

### Group Members

1. Ajith Kumar V (myself)
