# Cascade CNNs for Face Detection
The main idea is from 2015 CVPR Cascade CNNs for Face Detection.(L.Hao,Z.Lin etc.)
In the "master" branch, I include the code for training. And in the "test" branch, I include the code for testing.

#Install
1.Please install matconvernet first. You can get this library easily and visit the [homepage](http://www.vlfeat.org/matconvnet) to konw how to build.

2.After install matconvnet, just decompress the whole "test" branch files into the matconvnet folder.

3.Then modify simplenn.m in your matconvertnet/matlab/simplenn/simplenn.m.
Add two layers "custom" and "custom48" like what I write in my simplenn.m which has include in "test" branch.("master"branch also include simplenn.m but it is out of date.)
--note that you can alternatively replace it directly by my simplenn.m but I am not sure whether it is still compatible with the newest version of matconvnet.(because matconvnet updates quickly)

4.Now make sure you have compiled the matconvnet. Then you can easily start it by running the demo.m without recompile.
--If you have any questions, you could write an e-mail or open an issue to get contact with me. You are welcome.

#About Result
1.Speed: In fddb test, I use 16 different scales(scale factor:1.18) to resize the input so it's considerably slow. In real environment, you can change it to 8 different scales(scale factor:1.41) to speed up.(I have already make this change in the demo progamme) But unfortunately the speed is still about 3 or 4 seconds for a large pic. As far as I consider, the "for" in matlab might be the problem, I still wonder how to solve it.(Although I have used heatmap skill in the 12net and multi-thread tech) The advice is welcome.

2.Accuracy: The following picture is produced by 16 different scales input. The third pic is produced on fddb face detection test.

![](https://github.com/layumi/2015_Face_Detection/blob/test/test/p1597154992.jpg)
![](https://github.com/layumi/2015_Face_Detection/blob/test/show2.png)
![](https://github.com/layumi/2015_Face_Detection/blob/test/discROC-compare.png)

