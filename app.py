from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.lang import Builder
from kivy.uix.widget import Widget
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.clock import Clock
from kivy.graphics.texture import Texture

from kivy.core.window import Window
from kivy.properties import BooleanProperty

import os
import cv2
import numpy as np 

Builder.load_file('edge.kv')
Window.maximize() #剛打開就全螢幕
Window.clearcolor = (17/255, 30/255, 39/255, 1) #背景顏色

class ImageScreen(Screen):
    filter_choosed = "Canny" #預設Canny
    filter_buffer = "Canny"
    filename = ""
    Camera_isOpen = False #旗標，Camera是不是開著

    def press(self, filename): #選檔案
        if(filename):
            self.ids.my_image.source = filename[0]
            self.filename = filename[0]
            self.ids.my_image.reload()

    def show_image(self, style): #show圖片
        buf = cv2.flip(style, 0).tobytes()
        img_texture = Texture.create(size=(style.shape[1], style.shape[0]))
        img_texture.blit_buffer(buf, colorfmt='luminance', bufferfmt='ubyte')
        self.ids.Output_image.texture = img_texture

    def Camera_on(self): #按Use Camera
        if self.Camera_isOpen:
            pass
        else:
            self.capture = cv2.VideoCapture(1)
            Clock.schedule_interval(self.camera_input, 1.0/33.0) #開啟Timer
            Clock.schedule_interval(self.camera_output, 1.0/33.0)
            self.Camera_isOpen = True
    
    def Camera_off(self): #按Clear
        if self.Camera_isOpen:
            self.capture.release()
            Clock.unschedule(self.camera_input)
            Clock.unschedule(self.camera_output)
            self.Camera_isOpen = False

    def OutputImage(self): #按Submit來這(輸出在右邊)
        self.filter_choosed = self.filter_buffer
        self.slider_init()
        if self.Camera_isOpen == False:
            img = cv2.imread(self.filename)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_gray_blur = cv2.GaussianBlur(img_gray, (5,5), 0)

            ############      Canny      ############
            if self.filter_choosed == "Canny":    
                threshold1 = 50
                threshold2 = 100

                Canny = cv2.Canny(img_gray,threshold1,threshold2)
                self.show_image(Canny)

            ############      Laplacian      ############
            elif self.filter_choosed == "Laplacian":
                size = 5

                Laplacian_img = cv2.Laplacian(img_gray_blur, cv2.CV_64F, ksize=size)
                Laplacian_img = cv2.convertScaleAbs(Laplacian_img)
                # cv2.imshow('Laplacian_img', Laplacian_img)
                # cv2.imwrite('Laplacian.jpg', Laplacian_img)
                self.show_image(Laplacian_img)
                

            ############      Sobel      ############
            elif self.filter_choosed == "Sobel":
                sobelX = cv2.Sobel(img_gray_blur, cv2.CV_16S, 0, 1)
                sobelY = cv2.Sobel(img_gray_blur, cv2.CV_16S, 1, 0)
                sobelX = cv2.convertScaleAbs(sobelX)   # 转回uint8
                sobelY = cv2.convertScaleAbs(sobelY)
                sobel_mix = cv2.addWeighted(sobelX,0.5,sobelY,0.5,0)
                # cv2.imshow('sobelX', sobelX)
                # cv2.imwrite('sobelX.jpg', sobelX)
                # cv2.imshow('sobelY', sobelY)
                # cv2.imwrite('sobelY.jpg', sobelY)
                # cv2.imshow('sobel_mix', cv2.addWeighted(sobelX,0.5,sobelY,0.5,0))
                # cv2.imwrite('sobel_mix.jpg', cv2.addWeighted(sobelX,0.5,sobelY,0.5,0))
                self.show_image(sobel_mix)


            ############      Roberts      ############
            elif self.filter_choosed == "Roberts":
                unit = 10

                kernelx = np.array([[-unit,0],[0,unit]])
                kernely = np.array([[0,-unit],[unit,0]])
                x = cv2.filter2D(img_gray, cv2.CV_16S, kernelx)
                y = cv2.filter2D(img_gray, cv2.CV_16S, kernely)
                RobertsX = cv2.convertScaleAbs(x)
                RobertsY = cv2.convertScaleAbs(y)
                Roberts = cv2.addWeighted(RobertsX, 0.5, RobertsY, 0.5, 0)
                # cv2.imshow('RobertsX', RobertsX)
                # cv2.imwrite('RobertsX.jpg', RobertsX)
                # cv2.imshow('RobertsY', RobertsY)
                # cv2.imwrite('RobertsY.jpg', RobertsY)
                # cv2.imshow('Roberts_mix', Roberts)
                # cv2.imwrite('Roberts_mix.jpg', Roberts)
                self.show_image(Roberts)


            ############      Prewitt      ############
            elif self.filter_choosed == "Prewitt":
                unit = 3

                kernelX = np.array([[unit,unit,unit],[0,0,0],[-unit,-unit,-unit]])
                kernelY = np.array([[-unit,0,unit],[-unit,0,unit],[-unit,0,unit]])
                x = cv2.filter2D(img_gray, cv2.CV_16S, kernelX)
                y = cv2.filter2D(img_gray, cv2.CV_16S, kernelY)
                PrewittX = cv2.convertScaleAbs(x)
                PrewittY = cv2.convertScaleAbs(y)
                Prewitt_mix = cv2.addWeighted(PrewittX, 0.5, PrewittY, 0.5, 0)
                # cv2.imshow('PrewittX', PrewittX)
                # cv2.imwrite('PrewittX.jpg', PrewittX)
                # cv2.imshow('PrewittY', PrewittY)
                # cv2.imwrite('PrewittY.jpg', PrewittY)
                # cv2.imshow('Prewitt_mix', Prewitt_mix)
                # cv2.imwrite('Prewitt_mix.jpg', Prewitt_mix)
                self.show_image(Prewitt_mix)


            ############      Prewitt_Canny      ############
            elif self.filter_choosed == "Prewitt + Canny":
                unit = 3
                threshold1 = 50
                threshold2 = 100

                kernelX = np.array([[unit,unit,unit],[0,0,0],[-unit,-unit,-unit]])
                kernelY = np.array([[-unit,0,unit],[-unit,0,unit],[-unit,0,unit]])
                x = cv2.filter2D(img_gray, cv2.CV_16S, kernelX)
                y = cv2.filter2D(img_gray, cv2.CV_16S, kernelY)
                PrewittX = cv2.convertScaleAbs(x)
                PrewittY = cv2.convertScaleAbs(y)
                Prewitt_mix = cv2.addWeighted(PrewittX, 0.5, PrewittY, 0.5, 0)

                edges = cv2.Canny(img,threshold1,threshold2) 
                Prewitt_mix_edge_zero = np.bitwise_and(Prewitt_mix, np.bitwise_not(edges))

                (shape_x, shape_y) = edges.shape
                zeros = np.zeros((shape_x, shape_y), dtype=np.uint8)
                edges = cv2.addWeighted(edges, 0.7, zeros, 0.3, 0)

                Prewitt_Canny = np.bitwise_or(Prewitt_mix_edge_zero, edges)
                # cv2.imshow('Prewitt_Canny', Prewitt_Canny)
                # cv2.imwrite('Prewitt_Canny.jpg', Prewitt_Canny)
                self.show_image(Prewitt_Canny)
                

            ############      saliencyMap      ############
            elif self.filter_choosed == "SaliencyMap":
                saliency = cv2.saliency.StaticSaliencyFineGrained_create()
                (success, saliencyMap) = saliency.computeSaliency(img)
                saliencyMap = (saliencyMap * 255).astype("uint8")
                # cv2.imshow("saliencyMap_Grained", saliencyMap)
                # cv2.imwrite("saliencyMap_Grained.jpg", saliencyMap)
                # cv2.imshow("threshMap", threshMap)
                # cv2.imwrite("threshMap.jpg", threshMap)
                self.show_image(saliencyMap)
               
    def camera_input(self, *args): #Timer中斷，左邊輸出(Camera原畫面)
        ret, frame = self.capture.read()
        buf = cv2.flip(cv2.flip(frame, 0), 1).tobytes()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]))
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.ids.my_image.texture = img_texture

    def camera_output(self, *args): #Timer中斷，右邊輸出(Camera套濾鏡後)
        ret, frame = self.capture.read()
        frame = cv2.flip(frame, 1)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray_blur = cv2.GaussianBlur(frame_gray, (5,5), 0)

        threshold1 = self.ids.slider1.value
        threshold2 = self.ids.slider2.value
        threshold3 = self.ids.slider3.value

        ############      Canny      ############
        if self.filter_choosed == "Canny":
            Canny = cv2.Canny(frame_gray,threshold1,threshold2)
            self.show_image(Canny)

        ############      Laplacian      ############
        elif self.filter_choosed == "Laplacian":
            Laplacian_img = cv2.Laplacian(frame_gray_blur, cv2.CV_64F, ksize=threshold1)
            Laplacian_img = cv2.convertScaleAbs(Laplacian_img)
            # cv2.imshow('Laplacian_img', Laplacian_img)
            # cv2.imwrite('Laplacian.jpg', Laplacian_img)
            self.show_image(Laplacian_img)
            

        ############      Sobel      ############
        elif self.filter_choosed == "Sobel":
            sobelX = cv2.Sobel(frame_gray_blur, cv2.CV_16S, 0, 1)
            sobelY = cv2.Sobel(frame_gray_blur, cv2.CV_16S, 1, 0)
            sobelX = cv2.convertScaleAbs(sobelX)   # 转回uint8
            sobelY = cv2.convertScaleAbs(sobelY)
            sobel_mix = cv2.addWeighted(sobelX,threshold1,sobelY,threshold2,0)
            # cv2.imshow('sobelX', sobelX)
            # cv2.imwrite('sobelX.jpg', sobelX)
            # cv2.imshow('sobelY', sobelY)
            # cv2.imwrite('sobelY.jpg', sobelY)
            # cv2.imshow('sobel_mix', cv2.addWeighted(sobelX,0.5,sobelY,0.5,0))
            # cv2.imwrite('sobel_mix.jpg', cv2.addWeighted(sobelX,0.5,sobelY,0.5,0))
            self.show_image(sobel_mix)


        ############      Roberts      ############
        elif self.filter_choosed == "Roberts":

            kernelx = np.array([[-threshold1,0],[0,threshold1]])
            kernely = np.array([[0,-threshold1],[threshold1,0]])
            x = cv2.filter2D(frame_gray, cv2.CV_16S, kernelx)
            y = cv2.filter2D(frame_gray, cv2.CV_16S, kernely)
            RobertsX = cv2.convertScaleAbs(x)
            RobertsY = cv2.convertScaleAbs(y)
            Roberts = cv2.addWeighted(RobertsX, threshold1, RobertsY, threshold2, 0)
            # cv2.imshow('RobertsX', RobertsX)
            # cv2.imwrite('RobertsX.jpg', RobertsX)
            # cv2.imshow('RobertsY', RobertsY)
            # cv2.imwrite('RobertsY.jpg', RobertsY)
            # cv2.imshow('Roberts_mix', Roberts)
            # cv2.imwrite('Roberts_mix.jpg', Roberts)
            self.show_image(Roberts)


        ############      Prewitt      ############
        elif self.filter_choosed == "Prewitt":

            kernelX = np.array([[threshold1,threshold1,threshold1],[0,0,0],[-threshold1,-threshold1,-threshold1]])
            kernelY = np.array([[-threshold1,0,threshold1],[-threshold1,0,threshold1],[-threshold1,0,threshold1]])
            x = cv2.filter2D(frame_gray, cv2.CV_16S, kernelX)
            y = cv2.filter2D(frame_gray, cv2.CV_16S, kernelY)
            PrewittX = cv2.convertScaleAbs(x)
            PrewittY = cv2.convertScaleAbs(y)
            Prewitt_mix = cv2.addWeighted(PrewittX, threshold2, PrewittY, threshold3, 0)
            # cv2.imshow('PrewittX', PrewittX)
            # cv2.imwrite('PrewittX.jpg', PrewittX)
            # cv2.imshow('PrewittY', PrewittY)
            # cv2.imwrite('PrewittY.jpg', PrewittY)
            # cv2.imshow('Prewitt_mix', Prewitt_mix)
            # cv2.imwrite('Prewitt_mix.jpg', Prewitt_mix)
            self.show_image(Prewitt_mix)


        ############      Prewitt_Canny      ############
        elif self.filter_choosed == "Prewitt + Canny":
            print(threshold2 + threshold3)
            kernelX = np.array([[threshold1,threshold1,threshold1],[0,0,0],[-threshold1,-threshold1,-threshold1]])
            kernelY = np.array([[-threshold1,0,threshold1],[-threshold1,0,threshold1],[-threshold1,0,threshold1]])
            x = cv2.filter2D(frame_gray, cv2.CV_16S, kernelX)
            y = cv2.filter2D(frame_gray, cv2.CV_16S, kernelY)
            PrewittX = cv2.convertScaleAbs(x)
            PrewittY = cv2.convertScaleAbs(y)
            Prewitt_mix = cv2.addWeighted(PrewittX, 0.5, PrewittY, 0.5, 0)

            edges = cv2.Canny(frame,threshold2,threshold3) 
            Prewitt_mix_edge_zero = np.bitwise_and(Prewitt_mix, np.bitwise_not(edges))

            (shape_x, shape_y) = edges.shape
            zeros = np.zeros((shape_x, shape_y), dtype=np.uint8)
            edges = cv2.addWeighted(edges, 0.7, zeros, 0.3, 0)

            Prewitt_Canny = np.bitwise_or(Prewitt_mix_edge_zero, edges)
            # cv2.imshow('Prewitt_Canny', Prewitt_Canny)
            # cv2.imwrite('Prewitt_Canny.jpg', Prewitt_Canny)
            self.show_image(Prewitt_Canny)
            

        ############      saliencyMap      ############
        elif self.filter_choosed == "SaliencyMap":
            saliency = cv2.saliency.StaticSaliencyFineGrained_create()
            (success, saliencyMap) = saliency.computeSaliency(frame)
            saliencyMap = (saliencyMap * 255).astype("uint8")
            # cv2.imshow("saliencyMap_Grained", saliencyMap)
            # cv2.imwrite("saliencyMap_Grained.jpg", saliencyMap)
            # cv2.imshow("threshMap", threshMap)
            # cv2.imwrite("threshMap.jpg", threshMap)
            self.show_image(saliencyMap)
         
    def checkbox_click(self, instance, value, topping): #選擇濾鏡先把選的丟到buffer
        if value == True:
            self.filter_buffer = topping
        else:
            pass
    
    def slide(self, *args): #slider變動就來這裡
        threshold1 = self.ids.slider1.value
        threshold2 = self.ids.slider2.value
        threshold3 = self.ids.slider3.value
        self.ids.threshold1_value.text = str(round(self.ids.slider1.value, 2))
        self.ids.threshold2_value.text = str(round(self.ids.slider2.value, 2))
        self.ids.threshold3_value.text = str(round(self.ids.slider3.value, 2))

        if self.Camera_isOpen == False:
            img = cv2.imread(self.filename)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_gray_blur = cv2.GaussianBlur(img_gray, (5,5), 0)
            ############      Canny      ############
            if self.filter_choosed == "Canny":
                Canny = cv2.Canny(img_gray,threshold1,threshold2)
                self.show_image(Canny)

            ############      Laplacian      ############
            elif self.filter_choosed == "Laplacian":
                Laplacian_img = cv2.Laplacian(img_gray_blur, cv2.CV_64F, ksize=threshold1)
                Laplacian_img = cv2.convertScaleAbs(Laplacian_img)
                # cv2.imshow('Laplacian_img', Laplacian_img)
                # cv2.imwrite('Laplacian.jpg', Laplacian_img)
                self.show_image(Laplacian_img)
                

            ############      Sobel      ############
            elif self.filter_choosed == "Sobel":
                sobelX = cv2.Sobel(img_gray_blur, cv2.CV_16S, 0, 1)
                sobelY = cv2.Sobel(img_gray_blur, cv2.CV_16S, 1, 0)
                sobelX = cv2.convertScaleAbs(sobelX)   # 转回uint8
                sobelY = cv2.convertScaleAbs(sobelY)
                sobel_mix = cv2.addWeighted(sobelX,threshold1,sobelY,threshold2,0)
                # cv2.imshow('sobelX', sobelX)
                # cv2.imwrite('sobelX.jpg', sobelX)
                # cv2.imshow('sobelY', sobelY)
                # cv2.imwrite('sobelY.jpg', sobelY)
                # cv2.imshow('sobel_mix', cv2.addWeighted(sobelX,0.5,sobelY,0.5,0))
                # cv2.imwrite('sobel_mix.jpg', cv2.addWeighted(sobelX,0.5,sobelY,0.5,0))
                self.show_image(sobel_mix)


            ############      Roberts      ############
            elif self.filter_choosed == "Roberts":
                kernelx = np.array([[-threshold1,0],[0,threshold1]])
                kernely = np.array([[0,-threshold1],[threshold1,0]])
                x = cv2.filter2D(img_gray, cv2.CV_16S, kernelx)
                y = cv2.filter2D(img_gray, cv2.CV_16S, kernely)
                RobertsX = cv2.convertScaleAbs(x)
                RobertsY = cv2.convertScaleAbs(y)
                Roberts = cv2.addWeighted(RobertsX, threshold2, RobertsY, threshold3, 0)
                # cv2.imshow('RobertsX', RobertsX)
                # cv2.imwrite('RobertsX.jpg', RobertsX)
                # cv2.imshow('RobertsY', RobertsY)
                # cv2.imwrite('RobertsY.jpg', RobertsY)
                # cv2.imshow('Roberts_mix', Roberts)
                # cv2.imwrite('Roberts_mix.jpg', Roberts)
                self.show_image(Roberts)


            ############      Prewitt      ############
            elif self.filter_choosed == "Prewitt":

                kernelX = np.array([[threshold1,threshold1,threshold1],[0,0,0],[-threshold1,-threshold1,-threshold1]])
                kernelY = np.array([[-threshold1,0,threshold1],[-threshold1,0,threshold1],[-threshold1,0,threshold1]])
                x = cv2.filter2D(img_gray, cv2.CV_16S, kernelX)
                y = cv2.filter2D(img_gray, cv2.CV_16S, kernelY)
                PrewittX = cv2.convertScaleAbs(x)
                PrewittY = cv2.convertScaleAbs(y)
                Prewitt_mix = cv2.addWeighted(PrewittX, threshold2, PrewittY, threshold3, 0)
                # cv2.imshow('PrewittX', PrewittX)
                # cv2.imwrite('PrewittX.jpg', PrewittX)
                # cv2.imshow('PrewittY', PrewittY)
                # cv2.imwrite('PrewittY.jpg', PrewittY)
                # cv2.imshow('Prewitt_mix', Prewitt_mix)
                # cv2.imwrite('Prewitt_mix.jpg', Prewitt_mix)
                self.show_image(Prewitt_mix)


            ############      Prewitt_Canny      ############
            elif self.filter_choosed == "Prewitt + Canny":

                kernelX = np.array([[threshold1,threshold1,threshold1],[0,0,0],[-threshold1,-threshold1,-threshold1]])
                kernelY = np.array([[-threshold1,0,threshold1],[-threshold1,0,threshold1],[-threshold1,0,threshold1]])
                x = cv2.filter2D(img_gray, cv2.CV_16S, kernelX)
                y = cv2.filter2D(img_gray, cv2.CV_16S, kernelY)
                PrewittX = cv2.convertScaleAbs(x)
                PrewittY = cv2.convertScaleAbs(y)
                Prewitt_mix = cv2.addWeighted(PrewittX, 0.5, PrewittY, 0.5, 0)

                edges = cv2.Canny(img,threshold2,threshold3) 
                Prewitt_mix_edge_zero = np.bitwise_and(Prewitt_mix, np.bitwise_not(edges))

                (shape_x, shape_y) = edges.shape
                zeros = np.zeros((shape_x, shape_y), dtype=np.uint8)
                edges = cv2.addWeighted(edges, 0.7, zeros, 0.3, 0)

                Prewitt_Canny = np.bitwise_or(Prewitt_mix_edge_zero, edges)
                # cv2.imshow('Prewitt_Canny', Prewitt_Canny)
                # cv2.imwrite('Prewitt_Canny.jpg', Prewitt_Canny)
                self.show_image(Prewitt_Canny)
                

            ############      saliencyMap      ############
            elif self.filter_choosed == "SaliencyMap":
                saliency = cv2.saliency.StaticSaliencyFineGrained_create()
                (success, saliencyMap) = saliency.computeSaliency(img)
                saliencyMap = (saliencyMap * 255).astype("uint8")
                # cv2.imshow("saliencyMap_Grained", saliencyMap)
                # cv2.imwrite("saliencyMap_Grained.jpg", saliencyMap)
                # cv2.imshow("threshMap", threshMap)
                # cv2.imwrite("threshMap.jpg", threshMap)
                self.show_image(saliencyMap)
                
    def slider_init(self): #根據選擇濾鏡initial slider
        if self.filter_choosed == "Canny":
            self.ids.threshold1.text = "threshold1"
            self.ids.threshold1.opacity = 1
            self.ids.threshold1.disabled = 0
            self.ids.slider1.opacity = 1
            self.ids.slider1.disabled = 0
            self.ids.threshold1_value.opacity = 1
            self.ids.threshold1_value.disabled = 0

            self.ids.threshold2.text = "threshold2"
            self.ids.threshold2.opacity = 1
            self.ids.threshold2.disabled = 0
            self.ids.slider2.opacity = 1
            self.ids.slider2.disabled = 0
            self.ids.threshold2_value.opacity = 1
            self.ids.threshold2_value.disabled = 0

            self.ids.threshold3.opacity = 0
            self.ids.threshold3.disabled = 1
            self.ids.slider3.opacity = 0
            self.ids.slider3.disabled = 1
            self.ids.threshold3_value.opacity = 0
            self.ids.threshold3_value.disabled = 1
            
            self.ids.slider1.value = 50
            self.ids.slider1.min = 1
            self.ids.slider1.step = 1
            self.ids.slider1.max = 200

            self.ids.slider2.value = 100
            self.ids.slider2.min = 1
            self.ids.slider2.step = 1
            self.ids.slider2.max = 200
        elif self.filter_choosed == "Laplacian":
            self.ids.threshold1.text = "kernel_size"
            self.ids.threshold1.opacity = 1
            self.ids.threshold1.disabled = 0
            self.ids.slider1.opacity = 1
            self.ids.slider1.disabled = 0
            self.ids.threshold1_value.opacity = 1
            self.ids.threshold1_value.disabled = 0

            self.ids.threshold2.opacity = 0
            self.ids.threshold2.disabled = 1
            self.ids.slider2.opacity = 0
            self.ids.slider2.disabled = 1
            self.ids.threshold2_value.opacity = 0
            self.ids.threshold2_value.disabled = 1

            self.ids.threshold3.opacity = 0
            self.ids.threshold3.disabled = 1
            self.ids.slider3.opacity = 0
            self.ids.slider3.disabled = 1
            self.ids.threshold3_value.opacity = 0
            self.ids.threshold3_value.disabled = 1
            
            self.ids.slider1.value = 5
            self.ids.slider1.min = 1
            self.ids.slider1.step = 2
            self.ids.slider1.max = 9
        elif self.filter_choosed == "Sobel":
            self.ids.threshold1.text = "Sobel_X"
            self.ids.threshold1.opacity = 1
            self.ids.threshold1.disabled = 0
            self.ids.slider1.opacity = 1
            self.ids.slider1.disabled = 0
            self.ids.threshold1_value.opacity = 1
            self.ids.threshold1_value.disabled = 0

            self.ids.threshold2.text = "Sobel_Y"
            self.ids.threshold2.opacity = 1
            self.ids.threshold2.disabled = 0
            self.ids.slider2.opacity = 1
            self.ids.slider2.disabled = 0
            self.ids.threshold2_value.opacity = 1
            self.ids.threshold2_value.disabled = 0

            self.ids.threshold3.opacity = 0
            self.ids.threshold3.disabled = 1
            self.ids.slider3.opacity = 0
            self.ids.slider3.disabled = 1
            self.ids.threshold3_value.opacity = 0
            self.ids.threshold3_value.disabled = 1
            
            self.ids.slider1.value = 0.5
            self.ids.slider1.min = 0
            self.ids.slider1.step = 0.05
            self.ids.slider1.max = 1

            self.ids.slider2.value = 0.5
            self.ids.slider2.min = 0
            self.ids.slider2.step = 0.05
            self.ids.slider2.max = 1
        elif self.filter_choosed == "Roberts":
            self.ids.threshold1.text = "kernel_unit"
            self.ids.threshold1.opacity = 1
            self.ids.threshold1.disabled = 0
            self.ids.slider1.opacity = 1
            self.ids.slider1.disabled = 0
            self.ids.threshold1_value.opacity = 1
            self.ids.threshold1_value.disabled = 0

            self.ids.threshold2.text = "Roberts_X"
            self.ids.threshold2.opacity = 1
            self.ids.threshold2.disabled = 0
            self.ids.slider2.opacity = 1
            self.ids.slider2.disabled = 0
            self.ids.threshold2_value.opacity = 1
            self.ids.threshold2_value.disabled = 0

            self.ids.threshold3.text = "Roberts_Y"
            self.ids.threshold3.opacity = 1
            self.ids.threshold3.disabled = 0
            self.ids.slider3.opacity = 1
            self.ids.slider3.disabled = 0
            self.ids.threshold3_value.opacity = 1
            self.ids.threshold3_value.disabled = 0
            
            self.ids.slider1.value = 10
            self.ids.slider1.min = 1
            self.ids.slider1.step = 1
            self.ids.slider1.max = 30

            self.ids.slider2.value = 0.5
            self.ids.slider2.min = 0
            self.ids.slider2.step = 0.05
            self.ids.slider2.max = 1

            self.ids.slider3.value = 0.5
            self.ids.slider3.min = 0
            self.ids.slider3.step = 0.05
            self.ids.slider3.max = 1
        elif self.filter_choosed == "Prewitt":
            self.ids.threshold1.text = "kernel_unit"
            self.ids.threshold1.opacity = 1
            self.ids.threshold1.disabled = 0
            self.ids.slider1.opacity = 1
            self.ids.slider1.disabled = 0
            self.ids.threshold1_value.opacity = 1
            self.ids.threshold1_value.disabled = 0

            self.ids.threshold2.text = "Prewitt_X"
            self.ids.threshold2.opacity = 1
            self.ids.threshold2.disabled = 0
            self.ids.slider2.opacity = 1
            self.ids.slider2.disabled = 0
            self.ids.threshold2_value.opacity = 1
            self.ids.threshold2_value.disabled = 0

            self.ids.threshold3.text = "Prewitt_Y"
            self.ids.threshold3.opacity = 1
            self.ids.threshold3.disabled = 0
            self.ids.slider3.opacity = 1
            self.ids.slider3.disabled = 0
            self.ids.threshold3_value.opacity = 1
            self.ids.threshold3_value.disabled = 0
            
            self.ids.slider1.value = 3
            self.ids.slider1.min = 1
            self.ids.slider1.step = 1
            self.ids.slider1.max = 30

            self.ids.slider2.value = 0.5
            self.ids.slider2.min = 0
            self.ids.slider2.step = 0.05
            self.ids.slider2.max = 1

            self.ids.slider3.value = 0.5
            self.ids.slider3.min = 0
            self.ids.slider3.step = 0.05
            self.ids.slider3.max = 1
        elif self.filter_choosed == "Prewitt + Canny":
            self.ids.threshold1.text = "Prewitt_unit"
            self.ids.threshold1.opacity = 1
            self.ids.threshold1.disabled = 0
            self.ids.slider1.opacity = 1
            self.ids.slider1.disabled = 0
            self.ids.threshold1_value.opacity = 1
            self.ids.threshold1_value.disabled = 0

            self.ids.threshold2.text = "Canny_threshold1"
            self.ids.threshold2.opacity = 1
            self.ids.threshold2.disabled = 0
            self.ids.slider2.opacity = 1
            self.ids.slider2.disabled = 0
            self.ids.threshold2_value.opacity = 1
            self.ids.threshold2_value.disabled = 0

            self.ids.threshold3.text = "Canny_threshold2"
            self.ids.threshold3.opacity = 1
            self.ids.threshold3.disabled = 0
            self.ids.slider3.opacity = 1
            self.ids.slider3.disabled = 0
            self.ids.threshold3_value.opacity = 1
            self.ids.threshold3_value.disabled = 0

            self.ids.slider1.value = 3
            self.ids.slider1.min = 1
            self.ids.slider1.step = 1
            self.ids.slider1.max = 30

            self.ids.slider2.value = 50
            self.ids.slider2.min = 1
            self.ids.slider2.step = 1
            self.ids.slider2.max = 200

            self.ids.slider3.value = 100
            self.ids.slider3.min = 1
            self.ids.slider3.step = 1
            self.ids.slider3.max = 200
        elif self.filter_choosed == "SaliencyMap":
            self.ids.threshold1.opacity = 0
            self.ids.threshold1.disabled = 1
            self.ids.slider1.opacity = 0
            self.ids.slider1.disabled = 1
            self.ids.threshold1_value.opacity = 0
            self.ids.threshold1_value.disabled = 1

            self.ids.threshold2.opacity = 0
            self.ids.threshold2.disabled = 1
            self.ids.slider2.opacity = 0
            self.ids.slider2.disabled = 1
            self.ids.threshold2_value.opacity = 0
            self.ids.threshold2_value.disabled = 1

            self.ids.threshold3.opacity = 0
            self.ids.threshold3.disabled = 1
            self.ids.slider3.opacity = 0
            self.ids.slider3.disabled = 1
            self.ids.threshold3_value.opacity = 0
            self.ids.threshold3_value.disabled = 1
class SelectImage(Screen):
    pass

class MyApp(App):
    ShownImage = BooleanProperty(False)
    SubmitImage = BooleanProperty(False)
    def build(self):
        sm = ScreenManager()
        sm.add_widget(ImageScreen(name='image'))
        sm.add_widget(SelectImage(name='select'))
        return sm


if __name__ == '__main__':
    MyApp().run()