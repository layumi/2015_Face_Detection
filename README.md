# Cascade CNNs for Face Detection
The main idea is from 2015 CVPR Cascade CNNs for Face Detection.(L.Hao,Z.Lin etc.)
In the "master" branch, I include the code for training. And in the "test" branch, I include the code for testing.

#Install
1.Please install matconvernet first. You can get this library easily and visit the [homepage](http://www.vlfeat.org/matconvnet) to konw how to build.

2.After install matconvernet, you should modify simplenn.m in your matconvertnet/matlab/simplenn/simplenn.m.
Add two layers "custom" and "custom48" like what I write in my simplenn.m which has include in "test" branch.("master"branch also include simplenn.m but it is out of date.)
Now you can easily start it by running the demo.m without recompile.

![](https://github.com/layumi/2015_Face_Detection/blob/test/discROC-compare.png)

