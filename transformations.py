from PIL import Image
import random
import math
from PIL import ImageDraw

arrays = [[6, 2, 3, 1, 8, 0, 5, 4, 7],
 [0, 1, 2, 3, 4, 5, 6, 7, 8],
 [1, 0, 4, 2, 3, 6, 7, 8, 5],
 [2, 3, 0, 4, 1, 7, 8, 5, 6],
 [3, 4, 1, 5, 7, 8, 0, 6, 2],
 [4, 5, 7, 8, 6, 1, 2, 0, 3],
 [5, 6, 8, 7, 0, 2, 1, 3, 4],
 [7, 8, 5, 6, 2, 3, 4, 1, 0],
 [8, 7, 6, 0, 5, 4, 3, 2, 1],
 [0, 1, 2, 5, 6, 4, 7, 3, 8],
 [1, 0, 3, 6, 5, 7, 2, 8, 4],
 [2, 3, 0, 8, 7, 6, 4, 5, 1],
 [3, 2, 5, 0, 8, 1, 6, 4, 7],
 [4, 5, 6, 7, 0, 2, 8, 1, 3],
 [5, 4, 7, 2, 1, 8, 3, 6, 0],
 [6, 7, 8, 1, 4, 3, 5, 0, 2],
 [7, 8, 1, 4, 3, 5, 0, 2, 6],
 [8, 6, 4, 3, 2, 0, 1, 7, 5],
 [0, 1, 2, 6, 7, 4, 5, 3, 8],
 [1, 0, 3, 7, 5, 6, 4, 8, 2],
 [2, 3, 1, 0, 4, 5, 8, 6, 7],
 [3, 2, 0, 1, 6, 8, 7, 5, 4],
 [4, 5, 6, 8, 2, 7, 1, 0, 3],
 [5, 4, 7, 2, 8, 0, 3, 1, 6],
 [6, 8, 4, 3, 0, 1, 2, 7, 5],
 [8, 7, 5, 4, 1, 3, 6, 2, 0],
 [7, 6, 8, 5, 3, 2, 0, 4, 1],
 [0, 1, 2, 7, 5, 8, 4, 6, 3],
 [1, 0, 3, 6, 7, 4, 2, 8, 5],
 [2, 3, 1, 0, 6, 7, 5, 4, 8],
 [3, 2, 0, 1, 4, 5, 8, 7, 6],
 [4, 5, 7, 2, 8, 0, 6, 3, 1],
 [5, 4, 6, 8, 0, 3, 1, 2, 7],
 [6, 8, 4, 5, 3, 2, 7, 1, 0],
 [7, 6, 8, 3, 2, 1, 0, 5, 4],
 [8, 7, 5, 4, 1, 6, 3, 0, 2],
 [0, 1, 2, 8, 5, 3, 6, 7, 4],
 [1, 0, 4, 2, 3, 5, 7, 6, 8],
 [2, 3, 0, 7, 6, 8, 4, 5, 1],
 [3, 2, 1, 6, 7, 4, 5, 8, 0],
 [4, 5, 3, 0, 8, 7, 1, 2, 6],
 [5, 4, 6, 1, 2, 0, 8, 3, 7],
 [6, 8, 7, 3, 4, 2, 0, 1, 5],
 [7, 6, 8, 5, 0, 1, 2, 4, 3],
 [8, 7, 0, 4, 1, 6, 3, 5, 2],
 [1, 2, 5, 3, 8, 4, 7, 0, 6],
 [0, 1, 3, 2, 4, 7, 8, 6, 5],
 [2, 0, 1, 5, 7, 3, 6, 4, 8],
 [3, 4, 2, 6, 5, 8, 0, 1, 7],
 [4, 3, 8, 7, 6, 1, 5, 2, 0],
 [5, 6, 4, 8, 3, 0, 2, 7, 1],
 [6, 8, 7, 0, 2, 5, 1, 3, 4],
 [7, 5, 6, 1, 0, 2, 4, 8, 3],
 [1, 7, 5, 4, 8, 6, 3, 0, 2],
 [8, 0, 2, 4, 1, 5, 6, 3, 7],
 [0, 1, 4, 8, 7, 3, 5, 2, 6],
 [2, 5, 7, 3, 6, 4, 1, 8, 0],
 [4, 6, 3, 2, 5, 1, 0, 7, 8],
 [3, 4, 1, 0, 2, 7, 8, 6, 5],
 [5, 2, 8, 6, 3, 0, 7, 4, 1],
 [8, 3, 0, 7, 1, 6, 4, 5, 2],
 [6, 7, 5, 1, 4, 8, 2, 0, 3],
 [7, 8, 6, 5, 0, 2, 3, 1, 4],
 [0, 2, 7, 8, 3, 4, 1, 6, 5]]

class CropField(object):
    # crops he largest possible square  in the center of a picture
    def __init__(self):
        self.degree = 12

    def __call__(self, sample):
        width, height = sample.size
        new_size = min(width, height)
        n_left = 0
        n_right = width
        n_up = 0
        n_bottom = height
        dif_width = width-new_size
        dif_height = height-new_size
        dif_width/=2
        dif_height/=2

        n_left+=dif_width
        n_right-=dif_width

        n_up+=dif_height
        n_bottom-=dif_height
        sample = sample.crop((n_left, n_up, n_right, n_bottom))
        sample = sample.resize((224, 224), Image.LANCZOS)
        return sample

class ScaleChange(object):

    def __call__(self, sample):
        image_src = sample
        opcion = random.randint(1, 7)
        base = 0.9
        inc = 0.4
        #if(opcion == 7):
        ##    base = 0.55
        #    inc = 0.6
        zoom = random.random()*inc+base
        w, h = image_src.size
        w*=zoom
        h*=zoom
        w = int(w)
        h = int(h)
        image_src = image_src.resize((w, h), Image.LANCZOS)
        return image_src
class TileHerb(object):

    def __call__(self, sample):
        image_src = sample
        opcion = random.randint(1, 7)
        base = 0.9
        inc = 0.4
        #if(opcion == 7):
        ##    base = 0.55
        #    inc = 0.6
        zoom = random.random()*inc+base
        w, h = image_src.size
        w*=zoom
        h*=zoom
        w = int(w)
        h = int(h)
        image_src = image_src.resize((w, h), Image.LANCZOS)
        w, h = image_src.size
        punto_medio_x = w//2
        punto_medio_y = h//2
        if(opcion == 1):
            b = (punto_medio_x-(224//2), punto_medio_y-224, punto_medio_x+(224//2), punto_medio_y)
            ni = image_src.crop(box = b)
        if(opcion == 2):
            b = (punto_medio_x-(224//2), punto_medio_y, punto_medio_x+(224//2), punto_medio_y+224)
            ni = image_src.crop(box = b)

        #crop 3 4 5 6, un punto es el centro de la imagen, el otro es uno de los 4 lados
        if(opcion == 3):
            b = (punto_medio_x-224, punto_medio_y-224, punto_medio_x, punto_medio_y)
            ni = image_src.crop(box = b)

        if(opcion == 4):
            b = (punto_medio_x-224, punto_medio_y, punto_medio_x, punto_medio_y+224)
            ni = image_src.crop(box = b)
        if(opcion == 5):
            b = (punto_medio_x, punto_medio_y-224, punto_medio_x+224, punto_medio_y)
            ni = image_src.crop(box = b)

        if(opcion == 6):
            b = (punto_medio_x, punto_medio_y, punto_medio_x+224, punto_medio_y+224)
            ni = image_src.crop(box = b)

        if(opcion == 7): #crop en el centro
            left = (w - 224)/2
            top = (h - 224)/2
            right = (w + 224)/2
            bottom = (h + 224)/2

            # Crop the center of the image
            ni = image_src.crop((left, top, right, bottom))
            ni = image_src
        return ni

class TileCircle(object):

    def __call__(self, sample):
        image_src = sample
        opcion = random.randint(1, 7)
        base = 0.6
        inc = 0.8
        #if(opcion == 7):
        ##    base = 0.55
        #    inc = 0.6
        radio = 120
        zoom = random.random()*inc+base
        w, h = image_src.size
        w*=zoom
        h*=zoom
        w = int(w)
        h = int(h)
        image_src = image_src.resize((w, h), Image.LANCZOS)
        w, h = image_src.size
        punto_medio_x = w//2
        punto_medio_y = h//2
        coord_x = random.randint(-1*radio, radio)
        abs_x = abs(coord_x)
        abs_y = int(math.sqrt((radio*radio)-(abs_x*abs_x)))
        coord_y = random.randint(-1*abs_y, abs_y)
        nx = punto_medio_x+coord_x
        ny = punto_medio_y+coord_y

        left = (nx - 112)
        top = (ny - 112)
        right = (nx + 112)
        bottom = (ny + 112)

        # Crop the center of the image
        ni = image_src.crop((left, top, right, bottom))
        ni = image_src
        return ni

class AddBalldo(object):

    def __call__(self, image):
        pos = random.randint(1, 16)
        draw = ImageDraw.Draw(image)
        if(pos == 1):
            x = 28
            y = 28
        if(pos == 2):
            x = 84
            y = 28
        if(pos == 3):
            x = 140
            y = 28
        if(pos == 4):
            x = 196
            y = 28

        if(pos == 5):
            x = 28
            y = 84
        if(pos == 6):
            x = 84
            y = 84
        if(pos == 7):
            x = 140
            y = 84
        if(pos == 8):
            x = 196
            y = 84

        if(pos == 9):
            x = 28
            y = 140
        if(pos == 10):
            x = 84
            y = 140
        if(pos == 11):
            x = 140
            y = 140
        if(pos == 12):
            x = 196
            y = 140

        if(pos == 13):
            x = 28
            y = 196
        if(pos == 14):
            x = 84
            y = 196
        if(pos == 15):
            x = 140
            y = 196
        if(pos == 16):
            x = 196
            y = 196
        r = 24
        draw.ellipse((x-r, y-r, x+r, y+r), fill=(255,0,0,0))
        pos -= 1
        return image, pos

class AddJigsaw(object):

    def __call__(self, img):
        (imageWidth, imageHeight)=img.size
        gridx=3
        gridy=3
        rangex=int(imageWidth/gridx)
        rangey=int(imageHeight/gridy)
        new_image = Image.new('RGB', (224, 224), (255, 255, 255))
        arreglo = []
        coords = []
        for y in range(3):
            for x in range(3):
                base_x = rangex*x
                base_y = rangey*y
                bbox=(base_x, base_y, base_x+rangex, base_y+rangey)
                coords.append((base_x, base_y))
                slice_bit=img.crop(bbox)
                arreglo.append(slice_bit)
        opcion = random.randint(0, 63)

        new_order = arrays[opcion]
        opcion+=1
        for i in range(len(new_order)):
            img_temp = arreglo[new_order[i]]
            new_image.paste(img_temp, coords[i])
        return new_image, opcion

class AddRandomMancha(object):
    def __call__(self, image):
        draw = ImageDraw.Draw(image)
        manchar = random.randint(1, 2)
        if(manchar  == 2):
            r = 80
            x = 112
            y = 112
            draw.ellipse((x-r, y-r, x+r, y+r), fill=(0,0,0,0))
        manchar -= 1
        return image, manchar

"""
Convierte la imagen a blanco y negro
manchar: valor basura
"""
class AddBW(object):
    def __call__(self, image):
        image = image.convert("L")
        image = image.convert("RGB")
        manchar = 0
        return image, manchar

"""
Convierte la imagen a blanco y negro o a color con un 50% de prob
"""
class AddBWRandom(object):
    def __call__(self, image):
        opcion = random.randint(0, 1)
        if(opcion == 1):
            image = image.convert("L")
        manchar = 0
        return image, opcion


"""
Convierte la imagen a blanco y negro
manchar: valor basura
"""
class AddSepia(object):
    def sepia(self, img):
        width, height = img.size

        pixels = img.load() # create the pixel map

        for py in range(height):
            for px in range(width):
                r, g, b = img.getpixel((px, py))

                tr = int(0.393 * r + 0.769 * g + 0.189 * b)
                tg = int(0.349 * r + 0.686 * g + 0.168 * b)
                tb = int(0.272 * r + 0.534 * g + 0.131 * b)

                if tr > 255:
                    tr = 255

                if tg > 255:
                    tg = 255

                if tb > 255:
                    tb = 255

                pixels[px, py] = (tr,tg,tb)

        return img
    def __call__(self, image):
        image = self.sepia(image)
        manchar = 0
        return image, manchar


class AddRandomRotation(object):
    def __call__(self, image):
        opcion = random.randint(1, 8)
        angulo = 0
        if(opcion == 1):
            angulo = 0

        if(opcion == 2):
            angulo = 45

        if(opcion == 3):
            angulo = 90

        if(opcion == 4):
            angulo = 135
        if(opcion == 5):
            angulo = 180
        if(opcion == 6):
            angulo = 225
        if(opcion == 7):
            angulo = 270
        if(opcion == 8):
            angulo = 315
        image = image.rotate(angulo, expand = 1).resize((224, 224))
        opcion -= 1
        return image, opcion
