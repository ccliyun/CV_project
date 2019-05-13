# CV_project
  
**Requirement**  
python3  
numpy  
opencv-python  
skimage  
matplotlib  
cython  
  
**File List**  
Project1/                         -- directory of project 1  
dingbiao/                       -- directory of images used to camera calibration  
      *.jpg      
  test5.jpg                       -- testing image  
  dingbiao.py                    -- camera calibration program  
  my_pro.py                     -- height estimation program  
  
Project2/                         -- directory of project 2  
Classroom1-perfect/             -- classroom scene comes with perfect calibration   
  	calib.txt                      -- calibration information  
  	im{0,1}.png                   -- default left and right view  
  my_pro2.pyx                   -- functions used for depth calculation  
  setup.py                       -- setup program to compile my_pro2.pyx by cython  
  test.py                         -- testing program  
my_pro2.cpython-36m-x86_64-linux-gnu.so  
                               -- lib file which is the result of compiling  
  
**Running Steps**  
1, project 1: height caculaation form sigle view image  
`python3 dingbiao.py` for calibration  
`Python3 my_pro.py`   
2, project 2: depth map calculation using a correlation-based method  
`python3 setup.py build `   
(Alternative, if run, the result *.so should copy into the directory which test.py in)  
 `python3 test.py`  
