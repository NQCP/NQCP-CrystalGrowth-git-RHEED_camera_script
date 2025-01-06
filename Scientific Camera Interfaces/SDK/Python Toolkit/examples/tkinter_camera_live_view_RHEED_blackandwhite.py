import cv2 
import numpy as np
import time
import os 
import sys
import numba 

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


import threading
from PIL import Image, ImageTk

from scipy.stats import norm

try:
    #  For python 2.7 tkinter is named Tkinter
    import Tkinter as tk
except ImportError:
    import tkinter as tk
    
sys.path.append(r"C:\Users\qgh880\Desktop\Scientific_Camera_Interfaces_Windows-2.1\Scientific Camera Interfaces\SDK\Python Toolkit\thorlabs_tsi_camera_python_sdk_package\thorlabs_tsi_sdk-0.0.8")
sys.path.append(r"C:\Users\qgh880\Desktop\Scientific_Camera_Interfaces_Windows-2.1\Scientific Camera Interfaces\SDK\Python Toolkit\examples")

try:
    # if on Windows, use the provided setup script to add the DLLs folder to the PATH
    from windows_setup import configure_path
    configure_path()
except ImportError:
    configure_path = None

from thorlabs_tsi_sdk.tl_camera import TLCameraSDK #, TLCamera, Frame
#from thorlabs_tsi_sdk.tl_camera_enums import SENSOR_TYPE
#from thorlabs_tsi_sdk.tl_mono_to_color_processor import MonoToColorProcessorSDK

try:
    #  For Python 2.7 queue is named Queue
    import Queue as queue
except ImportError:
    import queue



class LiveViewCanvas(tk.Canvas):
    def gaussian_arr(self, n):
        size = 2 * n + 1; sigma=n/2
        x, y = np.meshgrid(np.arange(size), np.arange(size))
        distances = np.sqrt((x - n)**2 + (y - n)**2)
        gaussian_weights = norm.pdf(distances, loc=0, scale=sigma)
        gaussian_weights /= np.sum(gaussian_weights)
        return gaussian_weights
    
    def __init__(self, parent, image_queue, out, do_record, POI):
        # type: (typing.Any, queue.Queue) -> LiveViewCanvas
        self.image_queue = image_queue
        self.POI = POI
        self.POI_val = 0
        
        self.rectangle_id = None
        self.do_show_POI = False
        
        self._image_width = 0
        self._image_height = 0
        tk.Canvas.__init__(self, parent)
        self.pack()
        
        self.out = out
        self.do_record = do_record
        
        self.region_size = 5
        self.gauss_kernel = self.gaussian_arr(self.region_size)
        
        self.POI_vals = []
        self.POI_times = []
        self.store_POI_vals = False
        
        self._get_image()
        
    def _get_image(self):
        try:
            image = self.image_queue.get_nowait()
            
            if self.do_record:
                self.out.write(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
            
            self._image = ImageTk.PhotoImage(master=self, image=image)
            
            if (self._image.width() != self._image_width) or (self._image.height() != self._image_height):
                # resize the canvas to match the new image size
                self._image_width = 1000 #self._image.width()
                self._image_height = 800 #self._image.height()
                self.config(width=self._image_width, height=self._image_height)
            
            if self.do_show_POI == False:
                self.create_image(0, 0, image=self._image, anchor='nw')
            
            #if self.do_show_POI == True: #to be activated from Spot Tracker
            self.POI_val = image.getpixel((self.POI[1],self.POI[0]))
            half_w = 10
            
            if self.rectangle_id is not None:
                self.delete(self.rectangle_id)
            
            self.rectangle_id = self.create_rectangle(self.POI[1]-half_w,self.POI[0]-half_w, self.POI[1]+half_w,self.POI[0]+half_w, outline="blue", width=2)
            x,y = self.POI[0] , self.POI[1]
            
            self.POI_val = np.sum( np.array(image)[x-self.region_size:x+self.region_size+1, y-self.region_size:y+self.region_size+1]*self.gauss_kernel )
            
            if self.store_POI_vals:
                self.POI_vals.append(self.POI_val)
                self.POI_times.append(time.time())
                
            
        except queue.Empty:
            pass
        self.after(1, self._get_image) #!!!
    


class ImageAcquisitionThread(threading.Thread):

    def __init__(self, camera):
        # type: (TLCamera) -> ImageAcquisitionThread
        super(ImageAcquisitionThread, self).__init__()
        self._camera = camera
        self._previous_timestamp = 0
        
        self._is_color = False 
        self._bit_depth = camera.bit_depth
        self._camera.image_poll_timeout_ms = 0  # Do not want to block for long periods of time
        self._image_queue = queue.Queue(maxsize=1) #2
        self._stop_event = threading.Event()

    def get_output_queue(self):
        # type: (type(None)) -> queue.Queue
        return self._image_queue

    def stop(self):
        self._stop_event.set()


    def _get_image(self, frame):
        min_pixel_value = 0   # Adjust based on camera's typical dark level
        max_pixel_value = 255  # Adjust based on camera's typical bright level
    
        raw_image = frame.image_buffer
        
        scaled_image = np.clip(raw_image, min_pixel_value, max_pixel_value)  # Clip to fixed range
        scaled_image = (scaled_image - min_pixel_value) * (255 / (max_pixel_value - min_pixel_value))
        
        scaled_image = scaled_image.astype('uint8')  # Ensure 8-bit values
        return Image.fromarray(scaled_image, mode='L')  # Grayscale image
    
        
    def run(self):
        while not self._stop_event.is_set():
            try:
                frame = self._camera.get_pending_frame_or_null()
                if frame is not None:
                    if self._is_color:
                        pil_image = self._get_color_image(frame)
                        
                    else:
                        pil_image = self._get_image(frame)
                    self._image_queue.put_nowait(pil_image)
            except queue.Full:
                # No point in keeping this image around when the queue is full, let's skip to the next one
                pass
            except Exception as error:
                print("Encountered error: {error}, image acquisition will stop.".format(error=error))
                break
        print("Image acquisition has stopped")
        if self._is_color:
            self._mono_to_color_processor.dispose()
            self._mono_to_color_sdk.dispose()



class CameraFunc():
    def __init__(self,root):
        #super().__init__()
        self.image_acquisition_thread = None
        self.camera = None
        self.sdk = None
        self.root = None
        self.POI = (100,100)
        
        self.do_record = False
        self.out = None
        self.frame_rate = 30 #34 is max # Adjust the frame rate as needed
        
        
        self.start_cam()
        
    def start_cam(self):
        try:
            if self.camera_widget:
                try:
                    self.camera_widget.destroy()
                except tk.TclError:
                    pass
                else: 
                    self.camera.dispose()
                    self.sdk.dispose()
        except AttributeError:
            pass
        
        self.sdk = TLCameraSDK()
        camera_list = self.sdk.discover_available_cameras()
        
        self.camera = self.sdk.open_camera(camera_list[0])
        
        self.camera.is_frame_rate_control_enabled = True
        self.camera.frame_rate_control_value = self.frame_rate
        print('check framerate: ', self.frame_rate)
        
        self.camera.exposure_time_us = 50000 #50ms seems good
        
        print('frame rates: ',self.frame_rate, self.camera.frame_rate_control_value, self.camera.get_measured_frame_rate_fps())#34 is max
        if self.do_record:
            self.record()
            
            #self.out = cv2.VideoWriter(self.output_filename, self.fourcc, int(1000/self.frame_rate), (self.frame_width, self.frame_height))
            self.out = cv2.VideoWriter(self.output_filename, self.fourcc, 22, (self.frame_width, self.frame_height))
            #30 fps set gives ~22 fps into the cv2 writer
            
        # create generic Tk App with just a LiveViewCanvas widget
        print("Generating app...")
        self.root = root
        self.root.title(self.camera.name)
        self.image_acquisition_thread = ImageAcquisitionThread(self.camera)
        
        self.camera_widget = LiveViewCanvas(parent=self.root, image_queue=self.image_acquisition_thread.get_output_queue(), out = self.out, do_record=self.do_record, POI = self.POI)
        try:
            self.camera_widget.bind("<Button-1>", self.app.on_mouse_click)
        except AttributeError:
            pass
        print("Setting camera parameters...")
        self.camera.frames_per_trigger_zero_for_unlimited = 0
        self.camera.arm(2)
        self.camera.issue_software_trigger()
        
        print("Starting image acquisition thread...")
        self.image_acquisition_thread.start()
        
        print("App starting")
        
   
    
    def record(self):
        #self.output_path = r"C:\Users\qgh880\Desktop\Scientific_Camera_Interfaces_Windows-2.1\Scientific Camera Interfaces\SDK\Python Toolkit\examples"
        self.output_path = r"C:\Users\qgh880\Desktop\Videos"
        self.output_filename = self.output_path + "\Vid_" + str(self.app.record_name_entry.get())+"_"+time.strftime("%Y%m%d-%H%M%S")+".avi"
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can change the codec as needed
        self.frame_width = 1440  # Adjust the frame width as needed
        self.frame_height = 1080  # Adjust the frame height as needed
        #self.out = cv2.VideoWriter(self.output_filename, self.fourcc, self.frame_rate, (self.frame_width, self.frame_height))
        self.camera_widget.out = cv2.VideoWriter(self.output_filename, self.fourcc, int(1000/self.frame_rate), (self.frame_width, self.frame_height))
        
        self.do_record = True
        self.camera_widget.do_record = True
        
    def record_thread(self):
        self.rec_thread = threading.Thread(target=self.record)  # No parentheses for method reference
        self.rec_thread.start()
        
    def stop_record(self):
        self.camera_widget.do_record = False
        time.sleep(1)
        self.camera_widget.out.release()
        time.sleep(1)
        self.out = None
        self.camera_widget.out = None
        
    def start_cam_thread(self):
        self.cam_thread = threading.Thread(target=self.start_cam)
        self.cam_thread.start()
        
    def stop_cam_thread(self): 
        self.stop_cam_thr = threading.Thread(target=self.stop_cam)  # No parentheses for method reference
        self.stop_cam_thr.start()
        
    
    def stop_cam(self):
        if self.image_acquisition_thread:
            print("Waiting for image acquisition thread to finish...")
            self.image_acquisition_thread.stop()
            self.image_acquisition_thread.join()
        
        if self.camera:
            self.camera.disarm()
            self.camera.dispose()
            self.camera = None  # Reset the camera reference
        
        if self.sdk:
            self.sdk.dispose()
            #self.sdk = None  # Reset the SDK reference
            print('yes')
        if self.root:
            print("Closing resources...")
            
        try:
            self.camera_widget.out.release()
        except (NameError, AttributeError):
            print('No recording')
       
    def close_cam(self):
        try:
            self.stop_cam()
        except (NameError, AttributeError):
            print('already stopped')
        self.root.destroy()
        print('Bye bye')
        
class App:
    def __init__(self, _root, _cam):
        self.root = _root
        self.cam = _cam
        
        self.start_cam_button = tk.Button( self.root, text="Start cam" )
        self.start_cam_button.pack(side="left")
        
        self.start_cam_button.config(command=lambda: self.cam.start_cam_thread() )
        
        
        self.stop_cam_button = tk.Button( self.root, text="Stop cam" )
        self.stop_cam_button.pack(side="left")
        self.stop_cam_button.config(command=lambda: self.cam.stop_cam_thread() )
        
        self.ROI_button = tk.Button( self.root, text="ROI (does nothing atm)", command=self.use_ROI_fun )
        self.ROI_button.pack(side="left")
        
        self.POI_button = tk.Button( self.root, text="Show Feed", command=self.show_POI )
        self.POI_button.pack(side="left")
        
        self.new_window_button = tk.Button( self.root, text="Spot Int. Tracker", command=lambda: self.Spot_tracker() )
        self.new_window_button.pack(side="left")
        
        self.record_button = tk.Button( self.root, text="Record Feed: "+str(self.cam.do_record) )
        self.record_button.pack(side="left")
        self.record_button.config(command=lambda: self.record_switch() )
        
        self.record_name_label = tk.Label(self.root, text="Video name:")
        self.record_name_label.pack()
        self.record_name_entry = tk.Entry(self.root, width = 15)
        self.record_name_entry.pack()
        
        self.close_cam_button = tk.Button( self.root, text="close cam" )
        self.close_cam_button.pack(side="left")
        self.close_cam_button.config(command=lambda: self.cam.close_cam() )
        
        
        self.frame = tk.Frame(self.root)
        self.frame.pack()
        try:
            self.cam.camera_widget.bind("<Button-1>", self.on_mouse_click) #this tracks mouseclicks on the camerafeed
        except AttributeError:
            pass
        #self.cam.camera_widget.bind("<Button-1>", self.on_mouse_click2) #this tracks mouseclicks on the camerafeed
    def start_cam_app(self):
        self.root.destroy()
        root2 = tk.Tk()  
        camcam = CameraFunc(root2)     
        app = App(root, camcam)
        root2.mainloop()
        #self.__init__(self.root, self.cam)
    def record_switch(self):
        self.cam.do_record = not self.cam.do_record
        self.cam.camera_widget.do_record = not self.cam.camera_widget.do_record
        self.record_button.config(text="Rec: "+str(self.cam.do_record))
        #self.cam.record()
        print('self.cam.do_record = ', self.cam.do_record)
        print('self.cam.do_record = ', self.cam.camera_widget.do_record)
        
    def use_ROI_fun(self):
        pass
    def show_POI(self):
        self.cam.camera_widget.do_show_POI = not self.cam.camera_widget.do_show_POI
    def on_mouse_click(self, event):
        self.last_click = event
        x, y = event.x, event.y
        self.cam.camera_widget.POI = (y,x)
        try:
            val = self.cam.camera_widget.image_queue.get_nowait().getpixel((x,y))
            print('Pixel value: ', val)
        except queue.Empty: #self.cam.camera_widget._queue.Empty:
            pass
        
    def Spot_tracker(self): #plotting function
        def open_new_window():
            fig, ax = plt.subplots(figsize=(6, 4))
            
            live_curve_window = tk.Toplevel(root)
            live_curve_window.title("Spot Intensity Tracker")
            ax.set_title('RHEED Signal')
            
            canvas = FigureCanvasTkAgg(fig, master=live_curve_window)
            canvas.get_tk_widget().pack()
            self.counter = 0
            self.plot_start_time = time.time()
            self.fps = 30
            self.data_batch = 10
            
            def update_plot():
                if not paused:
                    self.x_data = np.array(self.cam.camera_widget.POI_times) - self.start_time
                    self.y_data = self.cam.camera_widget.POI_vals
                    ax.clear()
                    ax.plot(np.array(self.x_data), self.y_data,'.-')
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('Value')
                    ax.set_title('Live Curve')
                    canvas.draw()
            
                self.counter+=1
                live_curve_window.after(1000, update_plot)  # Update every 1000ms (1 second)
        
            paused = True
            ML_factor_  = 1
            n_cycles_ = 1
            self.ML_factor  = 1
            self.n_cycles = 1
            self.vertical_line_x = []
            
            def convert_to_ML(self):
                
                self.ML_factor  = float(self.conversion_entry.get() )
                nonlocal ML_factor_
                nonlocal n_cycles_
                ML_factor_ = float( self.conversion_entry.get() )
                n_cycles_ = float( self.cycle_entry.get() )
                self.distance_text.set(str(np.round(1/(self.vertical_line_x[1]-self.vertical_line_x[0])*ML_factor_*n_cycles_, 3) )  + ' ML/s' )
            
            def toggle_pause():
                nonlocal paused
                paused = not paused
                if not paused:
                    self.start_time = time.time()
                    self.cam.camera_widget.POI_vals = []
                    self.cam.camera_widget.POI_times = []
                    self.cam.camera_widget.store_POI_vals = True
                else:
                    self.cam.camera_widget.store_POI_vals = False
                 
            def clear():
                self.vertical_line_x = []
                ax.clear()
                ax.set_xlabel('Time')
                ax.set_ylabel('Value')
                ax.set_title('Live Data Plot')
                canvas.draw()
            
            def on_click(event):
                nonlocal ML_factor_
                
                self.vertical_line_x.append(event.xdata)#data
                if len(self.vertical_line_x)>1:
                    self.vertical_line_x=self.vertical_line_x[-2:]
                    self.distance_text.set(str((self.vertical_line_x[1]-self.vertical_line_x[0])*ML_factor_ ))
                if self.vertical_line_x:    
                    draw_vertical_line()
                    canvas.draw()    
            
            def draw_vertical_line():
                ax.clear()
                self.x_data = np.array(self.cam.camera_widget.POI_times) - self.start_time
                self.y_data = self.cam.camera_widget.POI_vals
                
                ax.plot(self.x_data, self.y_data,'.-')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Value')
                ax.set_title('Live Curve')
                if self.vertical_line_x:
                    for i in self.vertical_line_x:
                        ax.axvline(x=float(i), color='r', linestyle='--')
                
            def save_data():
                self.output_path = r"C:\Users\qgh880\Desktop\Videos"
                np.save(self.output_path + os.sep + str(self.name_entry.get())+"_"+time.strftime("%Y%m%d-%H%M%S")+'.npy', np.array([self.x_data,self.y_data]))
            def close():
                
                live_curve_window.destroy()
                
            update_plot() # Start updating the plot
            
            pause_button = tk.Button(live_curve_window, text="Run/Pause", command=toggle_pause)
            pause_button.pack()
            
            clear_button = tk.Button(live_curve_window, text="Clear", command=clear)
            clear_button.pack()
            
            conversion_label = tk.Label(live_curve_window, text="Write ML/s to um/hr: (1.018 for GaAs)")
            conversion_label.pack()
            self.conversion_entry = tk.Entry(live_curve_window, width = 10)
            self.conversion_entry.pack()
            
            self.cycle_label = tk.Label(live_curve_window, text="n RHEED Oscillations: ")
            self.cycle_label.pack()
            self.cycle_entry = tk.Entry(live_curve_window, width = 10)
            self.cycle_entry.pack()
            
            self.conversion_label = tk.Label(live_curve_window, textvariable='conversion') #self.distance_text)
            self.conversion_label.pack()
            conversion_button = tk.Button(live_curve_window, text="Get ML/s", command=lambda: convert_to_ML(self))
            conversion_button.pack()
            
            self.distance_text = tk.StringVar()
            self.distance_label = tk.Label(live_curve_window, textvariable=self.distance_text) #self.distance_text)
            self.distance_label.pack()
        
            write_file_label = tk.Label(live_curve_window, text='Write file name')
            write_file_label.pack()
            
            self.name_entry = tk.Entry(live_curve_window, width = 10)
            self.name_entry.pack()
            
            save_button = tk.Button(live_curve_window, text="Save Curve xy Data", command=save_data)
            save_button.pack()
            
            close_button = tk.Button(live_curve_window, text="Close", command=close) 
            close_button.pack()
            
            canvas.mpl_connect('button_press_event', on_click ) 
            
        open_new_window()
  
root = tk.Tk()  
camcam = CameraFunc(root)     
app = App(root, camcam)
camcam.app = app
root.mainloop()