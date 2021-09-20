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
    @IBOutlet weak var takePhotoButton: UIButton!
    @IBOutlet weak var capturedPreviewImage: UIImageView!
    @IBOutlet weak var boxesView: ShowBoxView!
    @IBOutlet weak var IoUSlider: UISlider!
    @IBOutlet weak var scoreSlider: UISlider!
   

    
    var binary_model = yolo() // myTrained()
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

    let color_mapping = [CGColor(red: 0.0, green: 0.7, blue: 1.0, alpha: 1.0),
                         CGColor(red: 1.0, green: 0.6, blue: 0.0, alpha: 1.0),
                         CGColor(red: 0.2, green: 1.0, blue: 0.0, alpha: 1.0),
                         CGColor(red: 0.0, green: 0.4, blue: 1.0, alpha: 1.0),
                         CGColor(red: 1.0, green: 0.3, blue: 0.0, alpha: 1.0),
                         CGColor(red: 0.8, green: 0.0, blue: 1.0, alpha: 1.0),
                         CGColor(red: 1.0, green: 0.0, blue: 0.3, alpha: 1.0),
                         CGColor(red: 0.8, green: 1.0, blue: 0.0, alpha: 1.0),
                         CGColor(red: 0.5, green: 0.0, blue: 1.0, alpha: 1.0),
                         CGColor(red: 0.0, green: 1.0, blue: 0.7, alpha: 1.0),
                         CGColor(red: 1.0, green: 0.9, blue: 0.0, alpha: 1.0),
                         CGColor(red: 0.0, green: 1.0, blue: 1.0, alpha: 1.0),
                         CGColor(red: 0.0, green: 0.1, blue: 1.0, alpha: 1.0),
                         CGColor(red: 0.0, green: 1.0, blue: 0.4, alpha: 1.0),
                         CGColor(red: 1.0, green: 0.0, blue: 0.9, alpha: 1.0),
                         CGColor(red: 0.0, green: 1.0, blue: 0.1, alpha: 1.0),
                         CGColor(red: 0.5, green: 1.0, blue: 0.0, alpha: 1.0),
                         CGColor(red: 0.2, green: 0.0, blue: 1.0, alpha: 1.0),
                         CGColor(red: 1.0, green: 0.0, blue: 0.6, alpha: 1.0),
                         CGColor(red: 1.0, green: 0.0, blue: 0.0, alpha: 1.0)]

    
    var result_ui_mapping = [NSInteger: UITextView]()
    var triggeredByCamera = false
    let resizeEnforce:CGFloat = 416
    let CONFINDENCE_INDEX = 4
    let CLASS_INDEX = 5
    let Y_MIN_INDEX = 0
    let X_MIN_INDEX = 1
    let Y_MAX_INDEX = 2
    let X_MAX_INDEX = 3
    var IoUThresholdValue:Float = 0.6
    var scoreThresholdValue:Float = 0.3

    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        
//        var my_model = YourNewModel()
//        print(my_mapping)
//        print(my_model)
//
//        let input_data = try? MLMultiArray(shape:[1, 64, 64, 3], dataType:.double)
//        print(input_data)
        // test code

        let my_image = UIImage(named: "images/try_mew2.jpg")
        print(my_image as Any)
        
//        let resized_image = resizeImage(image: my_image!, newWidth: 64, newHeight: 64)
//        print(resized_image)
//        let imageView = UIImageView(image: my_image)
//        imageView.frame = CGRect(x: 800m, y: 30, width: 150, height: 150)
//        self.view.addSubview(imageView)
        let my_prediction = doPredictFromImage(underPredictImage: my_image!)
        let filterd_result = yolo_predict_filter(predicted_result: my_prediction!, score_threshold: scoreThresholdValue, iou_threshold: IoUThresholdValue)
//         showInUI(myIdentity: my_prediction!)
        capturedPreviewImage.image = drawBoxesOnImage(image: my_image!, pboxes: filterd_result)
       
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
//        var showText = ""
//        var biggest = 0.0
//        var big_index = 0
//        for i in 0...(myIdentity.count)-1{
//            showText = String(format: "%.2f", Float(myIdentity[i])*100.0)+"% "+pokemon_mapping[i]!
//            result_ui_mapping[i]!.text = showText
//            if Double(myIdentity[i]) > biggest{
//                biggest = Double(myIdentity[i])
//                big_index = i
//            }
//        }
//        for i in 0...5 {
//            if i==big_index{
//                result_ui_mapping[i]!.textColor = UIColor.red
//            }
//            else{
//                result_ui_mapping[i]!.textColor = UIColor.black
//            }
//        }
    }
    
    func yolo_predict_filter(predicted_result: MLMultiArray, score_threshold: Float, iou_threshold: Float) -> Array<PredictedBox>{
        let predict_shape = predicted_result.shape
        var filtered_boxes: Array<PredictedBox> = Array()
        for grid_width in 0...predict_shape[1].intValue-1{
//            print(grid_width)
            for grid_height in 0...predict_shape[2].intValue-1{
                for anchor in 0...predict_shape[3].intValue-1 {
                    let confidence_key = [0, grid_width, grid_height, anchor, CONFINDENCE_INDEX] as [NSNumber]
                    let confidence_score = predicted_result[confidence_key]
                    if (Float(confidence_score) >= score_threshold){
//                        print("-----%f--", confidence_score)
                        let class_key = [0, grid_width, grid_height, anchor, CLASS_INDEX] as [NSNumber]
                        let class_type = predicted_result[class_key]
//                        print(classes_mapping[class_type.intValue])
                        let ymin_key = [0, grid_width, grid_height, anchor, Y_MIN_INDEX] as [NSNumber]
                        let xmin_key = [0, grid_width, grid_height, anchor, X_MIN_INDEX] as [NSNumber]
                        let ymax_key = [0, grid_width, grid_height, anchor, Y_MAX_INDEX] as [NSNumber]
                        let xmax_key = [0, grid_width, grid_height, anchor, X_MAX_INDEX] as [NSNumber]
                        let cur_box = PredictedBox()
                        cur_box.p_class = class_type.intValue
                        cur_box.p_score = Float(confidence_score)
                        cur_box.x_min = Float(predicted_result[xmin_key])
                        cur_box.y_min = Float(predicted_result[ymin_key])
                        cur_box.x_max = Float(predicted_result[xmax_key])
                        cur_box.y_max = Float(predicted_result[ymax_key])
                        print(confidence_score, class_type.intValue, Float(predicted_result[xmin_key])*924, Float(predicted_result[ymin_key])*1100, Float(predicted_result[xmax_key])*924, Float(predicted_result[ymax_key])*1100)
                        filtered_boxes.append(cur_box)
                    }
                }
            }
        }
        
        filtered_boxes = IOUFilter(boxArray: filtered_boxes, iou_threshold: iou_threshold)
        
        return filtered_boxes
    }
    
    func IOUFilter(boxArray: Array<PredictedBox>, iou_threshold: Float = 0.6, max_boxes: Int = 10)-> Array<PredictedBox>{
        var IOUed_boxes: Array<PredictedBox> = Array()
//        debugPrintScore(need_print: boxArray)
        let sortedArray = boxArray.sorted(by: {$0.p_score > $1.p_score})
//        debugPrintScore(need_print: sortedArray)
        
        for source_box in sortedArray {
            var over_IOU = false
            for target_box in IOUed_boxes {
                let iou_score = target_box.iouScore(compared_box: source_box)
                if iou_score > iou_threshold {
                    over_IOU = true
                    break
                }
            }
            
            // not over IOU, should be an object
            if !over_IOU {
                IOUed_boxes.append(source_box)
                if IOUed_boxes.count >= max_boxes {
                    break
                }
            }
        }
        
        
        return IOUed_boxes
    }
    
    func doPredictFromImage(underPredictImage: UIImage)-> MLMultiArray?{
        let resized_image = resizeImage(image: underPredictImage, newWidth: resizeEnforce, newHeight: resizeEnforce)
        var mlinputarray = resized_image.getPixelArray()
//        var test = resized_image.pixelBuffer(width: 64, height: 64)
        var input_value = yoloInput(input_33: mlinputarray)//myTrainedInput(input_1: mlinputarray)
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
        // dispose system shutter sound
        AudioServicesDisposeSystemSoundID(1108)
        print("auto capture triggered!")
        let settings = AVCapturePhotoSettings(format: [AVVideoCodecKey: AVVideoCodecType.jpeg])
        stillImageOutput.capturePhoto(with: settings, delegate: self)
    }
    @IBAction func IoUValueChanged(_ sender: Any) {
        IoUThresholdValue = IoUSlider.value
    }
    
    @IBAction func ScoreVauleChanged(_ sender: Any) {
        scoreThresholdValue = scoreSlider.value
    }
    
    @IBAction func didTakePhoto(_ sender: Any) {
        triggeredByCamera = true
        autoTakePhoto()
    }
    
    func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        // dispose system shutter sound
        AudioServicesDisposeSystemSoundID(1108)
        
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
        let filterd_result = yolo_predict_filter(predicted_result: my_prediction!, score_threshold: scoreThresholdValue, iou_threshold: IoUThresholdValue)
//         showInUI(myIdentity: my_prediction!)
        boxesView.drawBoxesOnMe(pboxes: filterd_result)
        showInUI(myIdentity: my_prediction!)
    }
    
    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        captureSession = AVCaptureSession()
        captureSession.sessionPreset = .high
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
    
    override func viewWillTransition(to size: CGSize, with coordinator: UIViewControllerTransitionCoordinator) {
        super.viewWillTransition(to: size, with: coordinator)
//        videoPreviewLayer.connection?.videoOrientation = UIOrientation_To_AVOrientation(ui: UIDevice.current.orientation)
    }
    
    func setupLivePreview() {
        
        videoPreviewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        
        videoPreviewLayer.videoGravity = .resizeAspectFill
        videoPreviewLayer.connection?.videoOrientation = .landscapeRight
            //UIOrientation_To_AVOrientation(ui: UIDevice.current.orientation)
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
    
    func drawBoxesOnImage(image: UIImage, pboxes: Array<PredictedBox>) -> UIImage {
        UIGraphicsBeginImageContext(image.size)
        let context = UIGraphicsGetCurrentContext()
        image.draw(at: CGPoint.zero)
        let textFont = UIFont(name: "Helvetica Bold", size: 40)!
        for cur_box in pboxes{
            let rectangle_show = cur_box.printRectangle(canvasSize: image.size)
            context!.setStrokeColor(color_mapping[cur_box.p_class])
            context!.setLineWidth(5)
            context!.addRect(rectangle_show)
            context!.drawPath(using: .stroke)
            let textFontAttributes = [
                NSAttributedString.Key.font: textFont,
                NSAttributedString.Key.foregroundColor: UIColor(cgColor: color_mapping[cur_box.p_class])
            ] as [NSAttributedString.Key : Any]
            let class_text = " " + cur_box.classText()
            class_text.draw(in: rectangle_show, withAttributes: textFontAttributes)
        }
        
        let newImage = UIGraphicsGetImageFromCurrentImageContext()!
        UIGraphicsEndImageContext()
        return newImage
    }

    func debugPrintScore(need_print: Array<PredictedBox>){
        print("Print array:", need_print)
        for element in need_print {
            print(element.p_score)
        }
    }

    func UIOrientation_To_AVOrientation(ui:UIDeviceOrientation)-> AVCaptureVideoOrientation{
        switch ui {
        case .landscapeLeft:        return .landscapeLeft
        case .landscapeRight:       return .landscapeRight
        case .portrait:             return .portrait
        case .portraitUpsideDown:   return .portraitUpsideDown
        default:                    return .portrait
        }
    }
}




