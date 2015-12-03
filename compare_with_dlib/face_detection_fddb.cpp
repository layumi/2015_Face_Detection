#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <iostream>

using namespace dlib;
using namespace std;

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv)
{  
    try
    {
        if (argc == 1)
        {
            cout << "Give some image files as arguments to this program." << endl;
            return 0;
        }

        frontal_face_detector detector = get_frontal_face_detector();
        //image_window win;
        int size=0;
        double x1=0,y1=0,x2=0,y2=0,w=0,h=0;
        // Loop over all the images provided on the command line.
        for (int i = 1; i < argc; ++i)
        {
            char prefix[100] = "/home/zzd/face-database/fddb/";
            prefix[29] = '\0';
            char jpgfix[] = ".jpg";
            char* location_tmp = strcat(prefix,argv[i]);
            char* location = strcat(location_tmp,jpgfix);  
            cout << argv[i] << endl;
            
            array2d<unsigned char> img;
            load_image(img, location);
            int old_scale = img.nc();
            pyramid_up(img);
            int new_scale = img.nc();
            double scale_factor = old_scale*1.0/new_scale;
            std::vector<rectangle> dets = detector(img);
            size = dets.size();
            cout << dets.size() << endl;
            for (int j=0; j<size; j++){
                x1 = dets[j].left();
                y1 = dets[j].top();
                x2 = dets[j].right();
                y2 = dets[j].bottom();
                w = x2 - x1;
                h = y2 - y1;
                y1 = y1 - h*0.2;
                h = y2 -y1;
            	cout<<(int)((x1+0.5)*scale_factor)<<"\t"<<(int)((y1+0.5)*scale_factor)<<"\t"<<(int)((w+0.5)*scale_factor)<<"\t"<<(int)((h+0.5)*scale_factor)<<"\t"<<1<<endl;
            }
            //win.clear_overlay();
            //win.set_image(img);
            //win.add_overlay(dets, rgb_pixel(255,0,0));
            //cout << "Hit enter to process the next image..." << endl;
            //cin.get();
        }
    }
    catch (exception& e)
    {
        cout << "\nexception thrown!" << endl;
        cout << e.what() << endl;
    }
}

// ----------------------------------------------------------------------------------------

