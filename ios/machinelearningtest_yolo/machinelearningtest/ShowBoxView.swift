//
//  ShowBoxView.swift
//  machinelearningtest
//
//  Created by Jimmy Sun on 8/24/21.
//  Copyright © 2021 Jimmy Sun. All rights reserved.
//

import Foundation
import UIKit

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

class ShowBoxView: UIView {
    var cur_pboxes: Array<PredictedBox>!
    let textFont = UIFont(name: "Helvetica Bold", size: 38)!
    
    override init(frame: CGRect){
        super.init(frame: frame)
    }
    
    required init?(coder: NSCoder) {
        super.init(coder: coder)
    }
    
    override func draw(_ rect: CGRect) {
        print(rect)
        
        if cur_pboxes != nil {
            print("there are ", cur_pboxes.count, "boxes")
            for cur_box in cur_pboxes{
                let rectangle_show = cur_box.printRectangle(canvasSize: rect.size)
                print (rectangle_show)
                let bpath:UIBezierPath = UIBezierPath(rect: rectangle_show)
                let color:UIColor = UIColor(cgColor: color_mapping[cur_box.p_class])
                color.set()
                bpath.stroke()
                let textFontAttributes = [
                    NSAttributedString.Key.font: textFont,
                    NSAttributedString.Key.foregroundColor: UIColor(cgColor: color_mapping[cur_box.p_class])
                ] as [NSAttributedString.Key : Any]
                let class_text = " " + cur_box.classText()
                class_text.draw(in: rectangle_show, withAttributes: textFontAttributes)
            }
            print("draw by predict")
        }
        else {
            let h = rect.height
            let w = rect.width
            let color:UIColor = UIColor.yellow

            let drect = CGRect(x: (w * 0.25),y: (h * 0.25),width: (w * 0.5),height: (h * 0.5))
            let bpath:UIBezierPath = UIBezierPath(rect: drect)

            color.set()
            bpath.stroke()

            print("it ran first time")
        }
    }
    
    func drawBoxesOnMe(pboxes: Array<PredictedBox>) {
        cur_pboxes = pboxes
        setNeedsDisplay()
    }
}
