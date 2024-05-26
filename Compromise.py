from PIL import Image
from PIL.ExifTags import TAGS
import os
folder = ["Birke", "Buche","Eiche","Kastanie","Linde","Kiefer","Kirsche", "Platane", "Robinie"]

def korrekte_orientierung(bild):
    try:
        exif = bild._getexif()
        if exif is not None:
            for tag, wert in exif.items():
                if TAGS.get(tag) == 'Orientation':
                    if wert == 3:
                        bild = bild.rotate(180, expand=True)
                    elif wert == 6:
                        bild = bild.rotate(270, expand=True)
                    elif wert == 8:
                        bild = bild.rotate(90, expand=True)
    except AttributeError:
        pass  # Bei Bildern ohne Exif-Daten wird eine AttributeError-Ausnahme ausgelöst
    return bild


def zuschneiden_bilder():
    for fo in folder:
        ordner_pfad ="./Trainingsdaten/" + fo
        destination= "./Komprimiert1/" + fo
        dateien = os.listdir(ordner_pfad)
        for datei in dateien:
            datei_pfad = os.path.join(ordner_pfad, datei)
            bild = Image.open(datei_pfad)
            bild = korrekte_orientierung(bild)
            zuschnitt = bild.resize((150, 200))
            breite, höhe = zuschnitt.size
            zuschnitt = zuschnitt.crop((25, 25, breite - 25, höhe - 25))
            zuschnitt_pfad = os.path.join((destination), '' + datei)
            zuschnitt.save(zuschnitt_pfad)

zuschneiden_bilder()