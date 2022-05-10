# CNN Real Time Face Detection

This program detects faces and distiguishes between me and not me.

Please note that I did not write the code fully by myself. I only took the original code from <a href="https://medium.com/@andimid">Dmytro Nikolaiev</a> which can be found <a href="https://towardsdatascience.com/how-to-create-real-time-face-detector-ff0e1f81925f/">here</a> and immproved the programme.


## Modifications/improvements to the original code

### Output Layer
Instead of using two neurons as output layer, one for the confidence that the face on the image is me and the other one for the confidence that it is not me, I used only one neuron with the sigmoid activation function. With this method, the network cannot have a high confidence for both classificatioin options at the same time. Therfore, rapid changes between the two classifications are reduced. If the network is unsure what to pick, it shows a percentage close to 50%.

### Face Cutout
The original code used to distort the images which were fed into the classifier network. This made it harder for the neural network to correctly identify the persons. My new version no longer distorts images and also improves the extension of the image when the face is close to the edge.

### Colour Gradient
To visualise when the network is unsure what to pick, the colour of the rectangle around the face changes to gray and the text no longer shows any label and confidence but solely "unsure".


## Dataset

To train the neural network to recognise you, you first have to collect images of you. To do so, you can use photos from your smartphone or extract frames from videos where your face is clearly visible. However, ideally you collect the data directly from your webcam to make it easier for the network to learn your prominent face characteristics. Preferably, pictures of people which are not you are also collected in this way. To do so, use the <a href="https://github.com/LevinHinder/CNN-Real-Time-Face-Detection/blob/main/collect%20data.py">collect_data.py</a> python script. Since it is very hard to have a diverse dataset with lots and lots of different people, I complemented my dataset with pictures from the <a href="https://github.com/NVlabs/ffhq-dataset">Flickr-Faces-HQ Dataset (FFHQ)</a> and the <a href="https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/">IMDB-WIKI Dataset</a>.

    Dataset
    ├── Me
    │   ├── img0.png
    │   ├── img1.png
    │   └── ...
    └── Not Me
        ├── img0.png
        ├── img1.png
        └── ...


## License

    MIT License

    Copyright (c) 2022 Levin Hinder

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
