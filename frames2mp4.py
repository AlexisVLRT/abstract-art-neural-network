import moviepy.editor as mp
import os

images = []
for i in range(len(os.listdir('frames')[:-1])):
    images.append('frames/out{}.png'.format(i))

clip = mp.ImageSequenceClip(images, fps=24)
print('Converting...')
clip.write_videofile("out2.mp4", threads=4, bitrate='5000k')
