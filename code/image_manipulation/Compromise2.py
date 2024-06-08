from PIL import Image
from PIL.ExifTags import TAGS
import os
# a list with our nine classes
folder = ["Birke", "Buche","Eiche","Kastanie","Linde","Kiefer","Kirsche", "Platane", "Robinie"]

# Prevent the images from being rotated when compressing.
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
        pass  # # images without  Exif-Data do not cause AttributeError-exception
    return bild


def zuschneiden_bilder():
    for fo in folder:
        ordner_pfad ="./train_data/" + fo
        destination= "./Komprimiert2/" + fo
        dateien = os.listdir(ordner_pfad)
        for datei in dateien:
            datei_pfad = os.path.join(ordner_pfad, datei)
            bild = Image.open(datei_pfad)
            bild = korrekte_orientierung(bild)
            breite, höhe = bild.size
            zuschnitt = bild.resize((int(breite/2), int(höhe/2)))
            breite, höhe = zuschnitt.size
            zuschnitt = zuschnitt.crop((breite/2-75, höhe/2-100, breite/2 + 75, höhe/2 + 100))
            zuschnitt_pfad = os.path.join((destination), '' + datei)
            zuschnitt.save(zuschnitt_pfad)

zuschneiden_bilder()