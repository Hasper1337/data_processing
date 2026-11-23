from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# https://upload.wikimedia.org/wikipedia/commons/8/86/StAnthony.jpg?utm_source=ru.wikipedia.org&utm_campaign=imageinfo&utm_content=original
im = Image.open('StAnthony.jpg')
#plt.imshow(im)
a = np.asarray(im)

target_color = np.array([253, 238, 147])
toler = 30

# a[50:70, 80:100] nado
#a[200:250, 70:120] ne nado
#plt.imshow(a[200:250, 70:120])

pos_region = a[50:70, 80:100]

neg_region = a[200:250, 70:120]

# Визуализация выбранных зон для проверки
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(pos_region)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(neg_region)
plt.axis('off')
plt.show()

pos_pixels = pos_region.reshape(-1, 3)
neg_pixels = neg_region.reshape(-1, 3)

df_pos = pd.DataFrame(pos_pixels, columns=['R', 'G', 'B'])
df_pos['Is it our color?'] = 1

df_neg = pd.DataFrame(neg_pixels, columns=['R', 'G', 'B'])
df_neg['Is it our color?'] = 0

df_train = pd.concat([df_pos, df_neg], ignore_index=True)

# перемешивание
df_train = df_train.sample(frac=1).reset_index(drop=True)

print("Размер обучающей выборки:", df_train.shape)
print(df_train[:50])
