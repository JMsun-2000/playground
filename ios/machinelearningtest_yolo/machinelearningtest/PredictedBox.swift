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
    var p_score:Float = 0.0
    var p_class = 0
    
    let classes_mapping = ["aeroplane",
                           "bicycle",
                           "bird",
                           "boat",
                           "bottle",
                           "bus",
                           "car",
                           "cat",
                           "chair",
                           "cow",
                           "diningtable",
                           "dog",
                           "horse",
                           "motorbike",
                           "person",
                           "pottedplant",
                           "sheep",
                           "sofa",
                           "train",
                           "tvmonitor"]
    
    func printRectangle(canvasSize:CGSize)->CGRect{
        let top_x = canvasSize.width * CGFloat(x_min)
        let top_y = canvasSize.height * CGFloat(y_min)
        let bottom_x = canvasSize.width * CGFloat(x_max)
        let bottom_y = canvasSize.height * CGFloat(y_max)
        let ret = CGRect(x: top_x, y: top_y, width: (bottom_x-top_x), height: (bottom_y-top_y))
        return ret
    }
    
    func classText()->String{
        return classes_mapping[p_class]
    }
    
    /*
     (-a, -b)
                  ┏━━━━━━━━━━━━━━━━┓
                  ┃                ┃
                  ┃  (-x, -y)      ┃
                  ┃      ┏━━━━━━━━━╋━━━━━━━┓
                  ┃      ┃  (a, b) ┃       ┃
                  ┗━━━━━━╋━━━━━━━━━┛       ┃
                         ┗━━━━━━━━━━━━━━━━━┛(x, y)
     */
    
    func iouScore(compared_box:PredictedBox)->Float{
        let xi1 = max(self.x_min, compared_box.x_min)
        let yi1 = max(self.y_min, compared_box.y_min)
        let xi2 = min(self.x_max, compared_box.x_max)
        let yi2 = min(self.y_max, compared_box.y_max)
        
        let inter_width = xi2-xi1
        let inter_height = yi2-yi1
        let inter_area = max(inter_height, 0) * max(inter_width, 0)
        
        let box1_area = (self.x_max - self.x_min)*(self.y_max - self.y_min)
        let box2_area = (compared_box.x_max - compared_box.x_min)*(compared_box.y_max - compared_box.y_min)
        let union_area = box1_area+box2_area-inter_area
        
        let iou = inter_area/union_area
        return iou
    }
}
