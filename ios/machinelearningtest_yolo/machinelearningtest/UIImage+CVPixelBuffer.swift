/*
  Copyright (c) 2017-2019 M.I. Hollemans

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to
  deal in the Software without restriction, including without limitation the
  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
  sell copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
  IN THE SOFTWARE.
*/

#if canImport(UIKit)

import UIKit
import VideoToolbox
import CoreML

extension UIImage {
  /**
    Resizes the image to width x height and converts it to an RGB CVPixelBuffer.
  */
  public func pixelBuffer(width: Int, height: Int) -> CVPixelBuffer? {
    return pixelBuffer(width: width, height: height,
                       pixelFormatType: kCVPixelFormatType_32ARGB,
                       colorSpace: CGColorSpaceCreateDeviceRGB(),
                       alphaInfo: .noneSkipFirst)
  }

  /**
    Resizes the image to width x height and converts it to a grayscale CVPixelBuffer.
  */
  public func pixelBufferGray(width: Int, height: Int) -> CVPixelBuffer? {
    return pixelBuffer(width: width, height: height,
                       pixelFormatType: kCVPixelFormatType_OneComponent8,
                       colorSpace: CGColorSpaceCreateDeviceGray(),
                       alphaInfo: .none)
  }

  func pixelBuffer(width: Int, height: Int, pixelFormatType: OSType,
                   colorSpace: CGColorSpace, alphaInfo: CGImageAlphaInfo) -> CVPixelBuffer? {
    var maybePixelBuffer: CVPixelBuffer?
    let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
                 kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue]
    let status = CVPixelBufferCreate(kCFAllocatorDefault,
                                     width,
                                     height,
                                     pixelFormatType,
                                     attrs as CFDictionary,
                                     &maybePixelBuffer)

    guard status == kCVReturnSuccess, let pixelBuffer = maybePixelBuffer else {
      return nil
    }

    let flags = CVPixelBufferLockFlags(rawValue: 0)
    guard kCVReturnSuccess == CVPixelBufferLockBaseAddress(pixelBuffer, flags) else {
      return nil
    }
    defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, flags) }

    guard let context = CGContext(data: CVPixelBufferGetBaseAddress(pixelBuffer),
                                  width: width,
                                  height: height,
                                  bitsPerComponent: 8,
                                  bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer),
                                  space: colorSpace,
                                  bitmapInfo: alphaInfo.rawValue)
    else {
      return nil
    }

    UIGraphicsPushContext(context)
    context.translateBy(x: 0, y: CGFloat(height))
    context.scaleBy(x: 1, y: -1)
    self.draw(in: CGRect(x: 0, y: 0, width: width, height: height))
    UIGraphicsPopContext()

    return pixelBuffer
  }
}
//On the top of your swift
extension UIImage {
    func getPixelColor(pos: CGPoint) -> Array<Any> {

        let pixelData = self.cgImage!.dataProvider!.data
        let data: UnsafePointer<UInt8> = CFDataGetBytePtr(pixelData)

        let pixelInfo: Int = ((Int(self.size.width) * Int(pos.y)) + Int(pos.x)) * 4
        print([data[0], data[1], data[2], data[3]])
        let test = 63*4
        print([data[test], data[test+1], data[test+2], data[test+3]])
        let b = CGFloat(data[pixelInfo]) / CGFloat(255.0)
        let g = CGFloat(data[pixelInfo+1]) / CGFloat(255.0)
        let r = CGFloat(data[pixelInfo+2]) / CGFloat(255.0)
        let a = CGFloat(data[pixelInfo+3]) / CGFloat(255.0)

        return [r, g, b]
    }
}

extension UIImage {
    func getPixelArray() -> MLMultiArray {

        let pixelData = self.cgImage!.dataProvider!.data
        let data: UnsafePointer<UInt8> = CFDataGetBytePtr(pixelData)
        let target_width = Int(self.size.width)
        let target_height = Int(self.size.height)
        
        var mlinputarray = try? MLMultiArray(shape: [1, NSNumber(value: target_height), NSNumber(value: target_width), 3], dataType: MLMultiArrayDataType.float32)
        
        for posY in 0...target_height-1 {
            for posX in 0...target_width-1 {
               let pos_inline = ((Int(target_width) * posY) + posX)
               let pixelInfo: Int = pos_inline * 4
               let b = Float(data[pixelInfo]) / Float(255.0)
               let g = Float(data[pixelInfo+1]) / Float(255.0)
               let r = Float(data[pixelInfo+2]) / Float(255.0)
//               print([r, g, b])
//                  print([data[pixelInfo+2], data[pixelInfo+1], data[pixelInfo]])
//               let targetpixel: Int = pos_inline * 3
//                mlinputarray![targetpixel] = NSNumber(value:r)
//                mlinputarray![targetpixel+1] = NSNumber(value:g)
//                mlinputarray![targetpixel+2] = NSNumber(value:b)
//                mlinputarray![pos_inline] = NSNumber(value:data[pixelInfo+2])
//                mlinputarray![pos_inline+1] = NSNumber(value:data[pixelInfo+1])
//                mlinputarray![pos_inline+2] = NSNumber(value:data[pixelInfo])
                mlinputarray?[[0, NSNumber(value:posY), NSNumber(value:posX), 0]]=NSNumber(value:r)
                mlinputarray?[[0, NSNumber(value:posY), NSNumber(value:posX), 1]]=NSNumber(value:g)
                mlinputarray?[[0, NSNumber(value:posY), NSNumber(value:posX), 2]]=NSNumber(value:b)
            }
        }
        print(mlinputarray![0], mlinputarray![1], mlinputarray![2])
        print(mlinputarray![3], mlinputarray![4], mlinputarray![5])
        return mlinputarray!
    }
}
//extension UIImage {
//  /**
//    Creates a new UIImage from a CVPixelBuffer.
//
//    - Note: Not all CVPixelBuffer pixel formats support conversion into a
//            CGImage-compatible pixel format.
//  */
//  public convenience init?(pixelBuffer: CVPixelBuffer) {
//    if let cgImage = CGImage.create(pixelBuffer: pixelBuffer) {
//      self.init(cgImage: cgImage)
//    } else {
//      return nil
//    }
//  }
//
//  /*
//  // Alternative implementation:
//  public convenience init?(pixelBuffer: CVPixelBuffer) {
//    // This converts the image to a CIImage first and then to a UIImage.
//    // Does not appear to work on the simulator but is OK on the device.
//    self.init(ciImage: CIImage(cvPixelBuffer: pixelBuffer))
//  }
//  */
//
//  /**
//    Creates a new UIImage from a CVPixelBuffer, using a Core Image context.
//  */
//  public convenience init?(pixelBuffer: CVPixelBuffer, context: CIContext) {
//    if let cgImage = CGImage.create(pixelBuffer: pixelBuffer, context: context) {
//      self.init(cgImage: cgImage)
//    } else {
//      return nil
//    }
//  }
//}

#endif
