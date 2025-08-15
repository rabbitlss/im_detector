import cv2
import json

class QuickLabeler:
    def __init__(self):
          self.annotations = []
          self.current_class = "chat_message"
          self.drawing = False
          self.start_point = None

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
              self.start_point = (x, y)
              self.drawing = True

        elif event == cv2.EVENT_LBUTTONUP:
              end_point = (x, y)
              self.drawing = False

              # 保存标注
              x1, y1 = self.start_point
              x2, y2 = end_point

              self.annotations.append({
                  "class": self.current_class,
                  "bbox": [min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2)]
              })

              print(f"Added: {self.current_class} at {self.annotations[-1]['bbox']}")

    def label_image(self, image_path):
        img = cv2.imread(image_path)
        cv2.namedWindow('Labeling')
        cv2.setMouseCallback('Labeling', self.mouse_callback)

        print("Instructions:")
        print("- Left click and drag to draw box")
        print("- Press 1-7 to change class")
        print("- Press S to save")
        print("- Press Q to quit")

        classes = ['receiver_avatar', 'receiver_name', 'input_box',
                    'send_button', 'chat_message', 'contact_item', 'user_avatar']

        while True:
            display = img.copy()

              # 绘制已有标注
            for ann in self.annotations:
                bbox = ann['bbox']
                cv2.rectangle(display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,255,0), 2)
                cv2.putText(display, ann['class'], (bbox[0], bbox[1]-5),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

              # 显示当前类别
            cv2.putText(display, f"Current: {self.current_class}", (10, 30),
                         cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

            cv2.imshow('Labeling', display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                  break
            elif key == ord('s'):
                  # 保存标注
                with open(image_path.replace('.jpg', '_labels.json'), 'w') as f:
                      json.dump({"objects": self.annotations}, f, indent=2)
                print("Saved!")
            elif ord('1') <= key <= ord('7'):
                  self.current_class = classes[key - ord('1')]
                  print(f"Switched to: {self.current_class}")

        cv2.destroyAllWindows()
        return self.annotations

  # 使用
labeler = QuickLabeler()
annotations = labeler.label_image("your_image.jpg")
