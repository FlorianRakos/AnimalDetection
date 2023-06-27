# Infos zur verwendung dieses Repositories

Dieses Repository ist abgeleitet von https://www.kaggle.com/code/nouamane/fine-tuning-detr-for-license-plates-detection/notebook?scriptVersionId=83102158
Es verwendet das pretrained detr-resnet-50 Object detection modell von facebook.

## Wie kann ein Modell trainiert werden?

1. Docker Container aufsetzen dafür wurde folgender verwendet:
    nvidia-docker run --gpus "device=0" -it --rm -v "$(pwd)"/AnimalDetection:/workspace/bambi nvcr.io/partners/gridai/pytorch-lightning:v1.4.0


2. Pip packages installieren
    -> siehe requirements.txt

3. Bild daten in /data/train und /data/val ablegen. Die dazugehörigen Annotations in /data/annotations als train.json und validation.json in Coco V1 format speichern.

4. In der Konsole python train.py ausführen. In der train.py kann die trainingsdauer (num_steps) sowie andere parameter festgelegt werden. Durch das hinzufügen des parameters -c (continue-training) kann von einem bestimmten checkpoint aus weiter trainiert werden. Dieser muss im folder /checkpoints/continueTraining hinterlegt werden. 
Um die Zwischenspeicherung von checkpoints zu konfigurieren kann man in der Detr Klasse die parameter save_top_k und every_n_epochs anpassen.
Mit save_top_k kann festgelegt werden, wie viele Modelle zwischengespeichert werden. Dabei werden die besten Modelle nach dem validation loss ausgewählt. Durch every_n_epochs definiert man wie oft dieser Zwischenspeicherungs Prozess läuft.
Damit man den Trainingsfortschritt auch verfolgen kann, wenn sich das terminal schließt ist es ratsam den konsolen output mit > ./runLogs/runX.txt in eine Textdatei zu speichern.

## Wie kann ein trainiertes Modell evaluiert werden?
Dies wird mit der evaluate.py im detr folder durchgeführt. Diese befindet sich im detr repository (github.com/facebookresearch/detr.git), da sie darin vorhandene libarys benötigt.
Wenn ein Modell evaluiert werden soll muss die .ckpt datei in den /checkpoint/load folder gelegt werden.
Wenn mehrere Modelle evaluiert werden sollen kann die evaluate.py mit dem Parameter -a (all) aufgerufen werden um alle checkpoints im load folder zu evaluieren.
Um die precision und recall werte besser auswerten zu können kann der konsolen output mit > evaluate.txt in ein Textdatei gespeichert werden.
Zur evaluierung des modelles sollten ebenfalls ein testdataset im /data/test vorhanden sein, sowie eine zugehörige test.json im annotations folder.

Zur evaluierung des Trainingsfortschrittes werden ebenfalls mit dem tensorboard package im folder checkpoints/default/lightning_logs trainingsinformationen gespeichert. Um diese Statistiken anzuzeigen muss man in den genannten folder navigieren und den befehl tensorboard --logdir=./ ausführen. Anschließen kann man in einem Webbrowser am genannten Port auf die Statistiken zugreifen.

## Wie kann ein trainiertes Modell benützt werden?
Um mit einem Modell (im load folder) Bilder zu annotieren wird ebenfalls die evaluate.py mit dem parameter -i (images) aufgerufen. Dabei wird momentan das testdataset verwendet. Als ergebniss werden die vom Modell annotierten bilder, sowie die ground truth gelabelten Bilder in dem results Ordner gespeichert.




