//
//  ViewController.swift
//  machinelearningtest
//
//  Created by Jimmy Sun on 2020/11/20.
//  Copyright © 2020 Jimmy Sun. All rights reserved.
//

import UIKit
import CoreML
import Vision
import AVFoundation


class ViewController: UIViewController, AVCapturePhotoCaptureDelegate {
    @IBOutlet weak var videoView: UIView!
    @IBOutlet weak var predictResultView0: UITextView!
    @IBOutlet weak var predictResultView1: UITextView!
    @IBOutlet weak var predictResultView2: UITextView!
    @IBOutlet weak var predictResultView3: UITextView!
    @IBOutlet weak var predictResultView4: UITextView!
    @IBOutlet weak var predictResultView5: UITextView!
    @IBOutlet weak var takePhotoButton: UIButton!
    @IBOutlet weak var capturedPreviewImage: UIImageView!
    @IBOutlet weak var samplepic0: UIImageView!
    @IBOutlet weak var samplepic1: UIImageView!
    @IBOutlet weak var samplepic2: UIImageView!
    @IBOutlet weak var samplepic3: UIImageView!
    @IBOutlet weak var samplepic4: UIImageView!
    @IBOutlet weak var samplepic5: UIImageView!
    
    var binary_model = myTrained()
    var captureSession: AVCaptureSession!
    var stillImageOutput: AVCapturePhotoOutput!
    var videoPreviewLayer: AVCaptureVideoPreviewLayer!
    var captureTimer: Timer?
    let my_mapping = [
        0: "zero: ",
        1: "one:  ",
        2: "two:  ",
        3: "three:",
        4: "four:  ",
        5: "five:  "
    ]
    let pokemon_mapping = [
        0: "bulbasaur: ",
        1: "charmander:  ",
        2: "squirtle:  ",
        3: "pikachu:",
        4: "lapras:  ",
        5: "mewtwo:  "
    ]
    var result_ui_mapping = [NSInteger: UITextView]()
    var triggeredByCamera = false
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
        result_ui_mapping = [
            0: predictResultView0,
            1: predictResultView1,
            2: predictResultView2,
            3: predictResultView3,
            4: predictResultView4,
            5: predictResultView5,
        ]
        
//        var my_model = YourNewModel()
//        print(my_mapping)
//        print(my_model)
//
//        let input_data = try? MLMultiArray(shape:[1, 64, 64, 3], dataType:.double)
//        print(input_data)
        // test code

        let my_image = UIImage(named: "images/try_mew2.jpg")
        print(my_image)
        capturedPreviewImage.image = my_image
//        let resized_image = resizeImage(image: my_image!, newWidth: 64, newHeight: 64)
//        print(resized_image)
//        let imageView = UIImageView(image: my_image)
//        imageView.frame = CGRect(x: 800, y: 30, width: 150, height: 150)
//        self.view.addSubview(imageView)
         let my_prediction = doPredictFromImage(underPredictImage: my_image!)
         showInUI(myIdentity: my_prediction!)
    
        samplepic0.image = UIImage(named: "images/no0.png")
        samplepic1.image = UIImage(named: "images/no1.png")
        samplepic2.image = UIImage(named: "images/no2.png")
        samplepic3.image = UIImage(named: "images/no3.png")
        samplepic4.image = UIImage(named: "images/no4.png")
        samplepic5.image = UIImage(named: "images/no5.png")
      /*
        let image_buffer = cvBuffer(from: resized_image)//cvBuffer(from: resized_image)
        //let image_buffer = my_image?.pixelBuffer(width: 64, height: 64)
        print(image_buffer)
        
        let input_test = YourNewModelInput(input_1: image_buffer!)
        let my_prediction = try? my_model.prediction(input: input_test)
        print(my_prediction?.Identity)
        
        
        */

        //-----------new here---------
//        var binary_model = myTrained()
//        var mlinputarray = resized_image.getPixelArray()
//        var input_value = myTrainedInput(input_1: mlinputarray)
//        let my_prediction = try? binary_model.prediction(input: input_value)
//        print(my_prediction?.Identity)

//        testMatrix()
    }
    
    func showInUI(myIdentity: MLMultiArray){
        var showText = ""
        var biggest = 0.0
        var big_index = 0
        for i in 0...(myIdentity.count)-1{
            showText = String(format: "%.2f", Float(myIdentity[i])*100.0)+"% "+pokemon_mapping[i]!
            result_ui_mapping[i]!.text = showText
            if Double(myIdentity[i]) > biggest{
                biggest = Double(myIdentity[i])
                big_index = i
            }
        }
        for i in 0...5 {
            if i==big_index{
                result_ui_mapping[i]!.textColor = UIColor.red
            }
            else{
                result_ui_mapping[i]!.textColor = UIColor.black
            }
        }
    }
    
    func doPredictFromImage(underPredictImage: UIImage)-> MLMultiArray?{
        let resized_image = resizeImage(image: underPredictImage, newWidth: 64, newHeight: 64)
        var mlinputarray = resized_image.getPixelArray()
//        var test = resized_image.pixelBuffer(width: 64, height: 64)
        var input_value = myTrainedInput(input_1: mlinputarray)
        let my_prediction = try? binary_model.prediction(input: input_value)
        print(my_prediction?.Identity)
        return my_prediction!.Identity
    }
    
    func testMatrix(){
        var mlinputarray = try? MLMultiArray(shape: [1, 5, 4, 3], dataType: MLMultiArrayDataType.int32)
        var numberInt = 1;
        for posY in 0...4 {
            for posX in 0...3 {
//                mlinputarray?[[0, NSNumber(value:posY), NSNumber(value:posX), 0]]=NSNumber(value:numberInt)
//                numberInt+=1
//                mlinputarray?[[0, NSNumber(value:posY), NSNumber(value:posX), 1]]=NSNumber(value:numberInt)
//                numberInt+=1
//                mlinputarray?[[0, NSNumber(value:posY), NSNumber(value:posX), 2]]=NSNumber(value:numberInt)
//                numberInt+=1
                let pos_inline = ((4 * posY) + posX)
                let targetpixel: Int = pos_inline * 3
                mlinputarray![pos_inline] = NSNumber(value:numberInt)
                numberInt+=1
                mlinputarray![pos_inline+1] = NSNumber(value:numberInt)
                numberInt+=1
                mlinputarray![pos_inline+2] = NSNumber(value:numberInt)
                numberInt+=1
            }
        }
        for post in 0...59 {
            print(mlinputarray![post])
        }
        
        
    }
    
    @objc func autoTakePhoto() {
        print("auto capture triggered!")
        let settings = AVCapturePhotoSettings(format: [AVVideoCodecKey: AVVideoCodecType.jpeg])
        stillImageOutput.capturePhoto(with: settings, delegate: self)
    }
    
    @IBAction func didTakePhoto(_ sender: Any) {
        triggeredByCamera = true
        autoTakePhoto()
    }
    
    func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        
        /* emulator hasn't camera  */
        guard let imageData = photo.fileDataRepresentation()
            else { return }
        
        let getimage = UIImage(data: imageData)

        // for test
//         let getimage =  UIImage(named: "images/try_mew2.jpg")
        if triggeredByCamera {
            capturedPreviewImage.image = getimage
            triggeredByCamera = false
        }
        let my_prediction = doPredictFromImage(underPredictImage: getimage!)
        showInUI(myIdentity: my_prediction!)
    }
    
    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        captureSession = AVCaptureSession()
        captureSession.sessionPreset = .medium
        guard let backCamera = AVCaptureDevice.default(for: AVMediaType.video)
            else {
                print("Unable to access back camera!")
                return
        }
        
        do {
            let input = try AVCaptureDeviceInput(device: backCamera)
            //Step 9
            stillImageOutput = AVCapturePhotoOutput()
            
            if captureSession.canAddInput(input) && captureSession.canAddOutput(stillImageOutput) {
                captureSession.addInput(input)
                captureSession.addOutput(stillImageOutput)
                setupLivePreview()
            }
        }
        catch let error  {
            print("Error Unable to initialize back camera:  \(error.localizedDescription)")
        }
        
        captureTimer = Timer.scheduledTimer(timeInterval: 1.0, target: self, selector: #selector(autoTakePhoto), userInfo: nil, repeats: true)
        
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        self.captureSession.stopRunning()
        self.captureTimer?.invalidate()
    }
    
    func setupLivePreview() {
        
        videoPreviewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        
        videoPreviewLayer.videoGravity = .resizeAspect
        videoPreviewLayer.connection?.videoOrientation = .portrait
        videoView.layer.addSublayer(videoPreviewLayer)
        
        //Step12
        DispatchQueue.global(qos: .userInitiated).async { //[weak self] in
            self.captureSession.startRunning()
            //Step 13
            DispatchQueue.main.async {
                self.videoPreviewLayer.frame = self.videoView.bounds
            }
        }
    }

    
    func cvBuffer(from image: UIImage) -> CVPixelBuffer? {
      let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue, kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue] as CFDictionary
      var pixelBuffer : CVPixelBuffer?
     let status = CVPixelBufferCreate(kCFAllocatorDefault, Int(image.size.width), Int(image.size.height), kCVPixelFormatType_32ARGB, attrs, &pixelBuffer)
     // let status = CVPixelBufferCreate(kCFAllocatorDefault, Int(image.size.width), Int(image.size.height), kCVPixelFormatType_32ARGB, attrs, &pixelBuffer)
      guard (status == kCVReturnSuccess) else {
        return nil
      }

      CVPixelBufferLockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))
      let pixelData = CVPixelBufferGetBaseAddress(pixelBuffer!)

      let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
      let context = CGContext(data: pixelData, width: Int(image.size.width), height: Int(image.size.height), bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer!), space: rgbColorSpace, bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)

      context?.translateBy(x: 0, y: image.size.height)
     context?.scaleBy(x: 1.0, y: -1.0)
      //context?.scaleBy(x: 1.0, y: 1.0)

      UIGraphicsPushContext(context!)
      image.draw(in: CGRect(x: 0, y: 0, width: image.size.width, height: image.size.height))
      UIGraphicsPopContext()
      CVPixelBufferUnlockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))

      return pixelBuffer
    }
    
    func myResultsMethod(request: VNRequest, error: Error?) {
        guard let results = request.results as? [VNClassificationObservation]
            else { fatalError("huh") }
        for classification in results {
            print(classification.identifier, // the scene label
                  classification.confidence)
        }

    }
    
    func resizeImage(image: UIImage, newWidth: CGFloat, newHeight: CGFloat) -> UIImage {
        UIGraphicsBeginImageContext(CGSize(width: newWidth, height: newHeight))
        image.draw(in: CGRect(x: 0, y: 0, width: newWidth, height: newHeight))
        let newImage = UIGraphicsGetImageFromCurrentImageContext()!
        UIGraphicsEndImageContext()

        return newImage
    }
//    public func preprocess(image: UIImage, width: Int, height: Int) -> MLMultiArray? {
//        let size = CGSize(width: width, height: height)
//
//
//        guard let pixels = try? image.pixelData().map({ (Double($0) / 255.0) }) else {
//            return nil
//        }
//
//        guard let array = try? MLMultiArray(shape: [3, height, width] as [NSNumber], dataType: .double) else {
//            return nil
//        }
//
//        let r = pixels.enumerated().filter { $0.offset % 4 == 0 }.map { $0.element }
//        let g = pixels.enumerated().filter { $0.offset % 4 == 1 }.map { $0.element }
//        let b = pixels.enumerated().filter { $0.offset % 4 == 2 }.map { $0.element }
//
//        let combination = r + g + b
//        for (index, element) in combination.enumerated() {
//            array[index] = NSNumber(value: element)
//        }
//
//        return array
//    }

}


