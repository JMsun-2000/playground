//
//  PredictedBox.swift
//  machinelearningtest
//
//  Created by Jimmy Sun on 8/20/21.
//  Copyright © 2021 Jimmy Sun. All rights reserved.
//

import Foundation
import UIKit

class PredictedBox {
    var y_min:Float = 0.0
    var x_min:Float = 0.0
    var y_max:Float = 1.0
    var x_max:Float = 1.0
    var p_scorce:Float = 0.0
    var p_class = 0
    
    func printRectangle(canvasSize:CGSize)->CGRect{
        let top_x = canvasSize.width * CGFloat(x_min)
        let top_y = canvasSize.height * CGFloat(y_min)
        let bottom_x = canvasSize.width * CGFloat(x_max)
        let bottom_y = canvasSize.height * CGFloat(y_max)
        let ret = CGRect(x: top_x, y: top_y, width: (bottom_x-top_x), height: (bottom_y-top_y))
        return ret
    }
}
