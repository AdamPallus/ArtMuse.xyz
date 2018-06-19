# ART MAGIC
Adam Pallus

This app is ready to load onto an AWS server. Testing has shown that it requires a t2.medium server to have enough memory to load the features matrix into memory.

##Use!

Upload a photo of a piece of art. The algorithm will suggest links to art for sale at fineartamerica.com that match the style and subject of your submission.

##How it works:
A convolutional neural network extracts abstract features from the image you upload and searches the feature-space for similar art. It will return art with an overall similar shape, a similar subject, or a similar painting style. You may find results matching any or all of the categories by scrolling down the page. 
