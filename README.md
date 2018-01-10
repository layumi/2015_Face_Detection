# Cascade CNNs for Face Detection
The main idea is from 2015 CVPR Cascade CNNs for Face Detection.(L.Hao,Z.Lin etc.)
In the "master" branch, I include the code for training. And in the "test" branch, I include the code for testing.(https://github.com/layumi/2015_Face_Detection/tree/test) 

What's New: I upload a short technical report in chinese. (https://github.com/layumi/2015_Face_Detection/blob/master/Short_Technical_Report_in_Chinese.pdf)

What's New: visualize net by https://github.com/layumi/visualize_face_detection_net/tree/master (Now for all net!)

What's New: visualize face and no-face data by feature cluster(extracted from 48net) with (https://github.com/layumi/bhtsne).
![](https://github.com/layumi/bhtsne/blob/master/face.jpg)

# Install
1.Please install matconvernet first. You can get this library easily and visit the [homepage](http://www.vlfeat.org/matconvnet) to konw how to build.

2.After install matconvnet, just decompress the whole "test" branch files into the matconvnet folder.

3.Then modify simplenn.m in your matconvertnet/matlab/simplenn/simplenn.m.
Add two layers "custom" and "custom48" like what I write in my simplenn.m which has been included in "test" branch.("master"branch also include simplenn.m but it is out of date.)
--note that you can alternatively replace it directly by my simplenn.m but I am not sure whether it is still compatible with the newest version of matconvnet.(because matconvnet updates quickly)

4.I wrote mex file to speed up the code. So then you may type 'mex zzd.c' to compile the c file I included as well. 

5.Then you can easily start it by running the demo.m. And add the pic which you like in the picture folder.

--If you have any questions, you could write an e-mail or open an issue to get contact with me. You are welcome.

# About Result
1.Speed: In fddb test, I use 16 different scales(scale factor:1.18) to resize the input so it's considerably slow. In real environment, you can change it to 8 different scales(scale factor:1.41) to speed up.(I have already make this change in the demo progamme) But unfortunately the speed is still about 3 or 4 seconds for a large pic. As far as I consider, the "for" in matlab might be the problem, I still wonder how to solve it.(Although I have used heatmap skill in the 12net and multi-thread tech) The advice is welcome.

Tips: In the newest version, I add zzd.c which is a c file to get pic patches. By avoid using matlab "for", it saves 50% time and have a better effieciency.  I have included mex executable file, but it is better to recompile it for your own environment by "mex zzd.c". 

2.Accuracy: The following picture is produced by 16 different scales input. The third pic is produced on fddb face detection test.(As the original paper said, I enlarge the bounding boxes by y1 = y1-(y2-y1)*0.4 while testing in fddb)

![](https://github.com/layumi/2015_Face_Detection/blob/test/test/p1597154992.jpg)
![](https://github.com/layumi/2015_Face_Detection/blob/test/show2.png)
![](https://github.com/layumi/2015_Face_Detection/blob/test/discROC-compare.png)


# Citation
We greatly appreciate it if you can cite the website in your publications:
```
@misc{2015_Face_Detection,
  title = {{2015_Face_Detection}},
  howpublished = "\url{https://github.com/layumi/2015_Face_Detection}",
}
```
