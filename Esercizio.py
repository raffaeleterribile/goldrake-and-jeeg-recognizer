# Impostazioni del notebook per ricaricare automaticamente le librerie in caso di modifiche
#%reload_ext autoreload
#%autoreload 2
#%matplotlib inline

# Configurazione fastai e corso
# Questo è il contenuto dello script Bash
#	#!/bin/bash
#	if [ ! -e /content/models ]; then
#	        mkdir -p /root/.torch/models
#	        mkdir -p /root/.fastai/data
#	        ln -s /root/.torch/models /content
#	        ln -s /root/.fastai/data /content
#	        rm -rf /content/sample_data/
#	fi
#	
#	echo Updating fastai...
#	pip install fastai --upgrade > /dev/null
#	echo Done.
#!curl -s https://course.fast.ai/setup/colab | bash

# Script da eseguire dopo una ricerca di immagini in Google
# Questo script scarica in un file gli url delle immagini trovate

#urls = Array.from(document.querySelectorAll('.rg_di .rg_meta')).map(el=>JSON.parse(el.textContent).ou);
#window.open('data:text/csv;charset=utf-8,' + escape(urls.join('\n')));

# Dopo aver scaricato i file, caricarli manualmente sul server
# Le modalità di caricamento dipendo dal sistema che si sta utilizzando
# Con Google Colaboratory occorre aprire la barra laterale a sinistra,
# andare nella sezione "Files" ed usare il pulsante "Upload" o fare clic con il tasto destro su una cartella e selezionare "Upload"

from pathlib import Path

# Creazione delle cartelle per i dati
current_path = Path.cwd()

data_path = current_path / 'data'
data_path.mkdir(parents = True, exist_ok = True)

models_path = current_path / 'models'
models_path.mkdir(parents = True, exist_ok = True)

daitarn3_path = data_path / 'daitarn3'
daitarn3_path.mkdir(parents = True, exist_ok = True)

goldrake_path = data_path / 'goldrake'
goldrake_path.mkdir(parents = True, exist_ok = True)

jeeg_path = data_path / 'jeeg'
jeeg_path.mkdir(parents = True, exist_ok = True)

from fastai.vision import *

# Scaricamento delle immagini
# Si usa la funzione 'download_images' che prende come parametri un file (o un insieme) di url e la cartella in cui salvare le immagini
#download_images(current_path / (daitarn3_path.name + '.txt'), daitarn3_path)
#download_images(current_path / (goldrake_path.name + '.txt'), goldrake_path)
#download_images(current_path / (jeeg_path.name + '.txt'), jeeg_path)

# Verifica delle immagini scaricate: quelle errate vengono eliminate, le altre vengono ridimensionate
# Si usa la funzione 'verify_images' che prende in input una cartella ed esamina tutti i file presenti
#for directory in data_path.iterdir():
#	if directory.is_dir():
#		verify_images(directory, delete = True, max_size = 500)

# Si definisce una funzione che imposta i seed a partire dai quali calcolare i valori casuali
def set_random_seed(seed):
	# python RNG
	random.seed(seed)

	# pytorch RNGs
	import torch
	torch.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)

	# numpy RNG
	import numpy as np
	np.random.seed(seed)

set_random_seed(0)

# Si carica il DataSet, creando le etichette dei dati a partire dai nomi delle cartelle e dividendo il DataSet in Train Set e Validation Set in base ad una percentuale
# I parametri 'ds_tfms' e 'num_workers' non sono indispensabili
# Per ottenere risultati riproducibili, oltre ad impostare i seed, occorre impostare num_workers = 1 o = 0
data = ImageDataBunch.from_folder(data_path, train = data_path, valid_pct = 0.2, \
	ds_tfms = get_transforms(), size = 224, num_workers = 0).normalize(imagenet_stats)

# Si visualizzano un po' di immagini di esempio
#data.show_batch()

# Qualche statistica sui dati
print(data.classes)
print(data.c)
print(len(data.train_ds))
print(len(data.valid_ds))

# Creazione del modello
learner = cnn_learner(data, models.resnet34, metrics = [accuracy, error_rate])

# Prima analisi
learner.fit_one_cycle(2, max_lr = slice(1e-4, 1e-3))

# Salvataggio del modello
learner.save(models_path / 'model-01')

# Sblocco dei parametri
learner.unfreeze()

# Ricerca del Learning Rate migliore
learner.lr_find()

# Visualizzazione dei risultati della ricerca
learner.recorder.plot()

# Ripristino del modello salvato
learner.load(models_path / 'model-01')

# Analisi con il Learning Rate trovato
learner.fit_one_cycle(4, max_lr = slice(1e-4, 1e-3))

# Salvataggio del modello
learner.save(models_path / 'model-02')

# Crezione della classe per l'interpretazione dei risultati
interpreter = ClassificationInterpretation.from_learner(learner)

# Visualizzazione della matrice di confusione
interpreter.plot_confusion_matrix()

# Visualizzazione degli errori più gravi
interpreter.plot_top_losses(12)

# Elenco delle classi che danno più problemi
interpreter.most_confused(min_val = 2)

# E' possibile che alcuni errori siano causati da errori nel DataSet iniziale (immagini classificate male)
# Per verificarlo, si utilizza un widget

from fastai.widgets import *

# Creazione di un DataSet contenente tutte le immagini
images = ImageList.from_folder(data_path).split_none().label_from_folder().transform(get_transforms(), size = 224).databunch()

# Si crea un modello simile al precedente, ma che non suddivide i dati in Train Set e Validation Set
cleaner = cnn_learner(images, models.resnet34, metrics = [accuracy, error_rate])

# Si carica il modello salvato
cleaner.load(models_path / 'model-02')

# Si estraggono le immagini (ed i loro indici) che hanno dato i maggiori errori
dataset, indexes = DatasetFormatter().from_toplosses(cleaner)

# Si visualizzano le immagini e, per ognuna, si decide se eliminarla o se cambiarne l'etichetta
# In realtà le immagini originali non vengono toccate minimamente: viene solo creato un file chiamato 'cleaned.csv' contenente le operazioni eseguite
# Successivamente sarà possibile ricreare il DataSet a partire da questo file usando il metodo 'ImageDataBunch.from_csv' e riaddestrare il modello
ImageCleaner(dataset, indexes, data_path)

# E' possibile anche visualizzare le immagini più simili, in modo da eliminare manualmente i duplicati
dataset, indexes = DatasetFormatter().from_similars(cleaner)

# Come prima, le operazioni vengono salvate in 'cleaned.csv'
ImageCleaner(dataset, indexes, data_path, duplicates = True)

# Per il passaggio in produzione occorre esportare il modello in un formato contenente i parametri ed informazioni sulle classi, sulle trasformazioni applicate alle immagini
# Viene creato un file chiamato 'export.pkl'
exported_path = models_path / 'export.pkl'
learner.export(exported_path)

# In produzione conviene usare la CPU invece della GPU perché quasi sicuramente si elaborerà un'immagine per volta
defaults.device = torch.device('cpu')

# Si carica un'immagine manualmente
image_path = current_path / 'image.jpg'

# Si carica l'immagin in memoria
image = open_image(image_path)
image

# Si carica il modello esportato
learner = load_learner(models_path)

# Si esamina l'immagine
predicted_class, predicted_index, outputs = learner.predict(image)
print(predicted_class)
print(predicted_index)
print(outputs)
