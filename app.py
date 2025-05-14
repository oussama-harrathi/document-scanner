#.\venv\Scripts\Activate.ps1 
#python app.py test3.jpg 
#python app.py test2.jpg --interactive

import cv2
import numpy as np
import argparse
import os
from datetime import datetime

class DocumentScanner:
    def __init__(self):
        self.orig_image = None       
        self.image_for_contour = None 
        self.gray_for_contour = None 
        self.loaded_orig_path = None 
        self.output_dir = "scanned_documents"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def load_original_image(self, image_path):
        """Loads the pristine original image."""
        self.orig_image = cv2.imread(image_path)
        if self.orig_image is None:
            raise Exception(f"Could not read image: {image_path}")
        self.loaded_orig_path = image_path
        return self.orig_image

    def preprocess_image_for_contour(self, image_to_preprocess_bgr):
        """
        Prepares a BGR image specifically for contour detection if needed.
        Returns a BGR image.
        """
        gray = cv2.cvtColor(image_to_preprocess_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 30, 100)
        kernel = np.ones((5, 5), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)
        
        result_bgr = image_to_preprocess_bgr.copy() 

        contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            largest_contour = contours[0]
            
            # Create a mask from the largest contour
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [largest_contour], 0, 255, -1)
            mask = cv2.GaussianBlur(mask, (5, 5), 0) # Soften mask edges
            
            # Enhance contrast using CLAHE on L channel of LAB color space
            lab = cv2.cvtColor(image_to_preprocess_bgr, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l)
            enhanced_lab = cv2.merge((l_enhanced, a, b))
            enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            
            
            mask_3channel = cv2.merge([mask, mask, mask])
            mask_3channel_float = mask_3channel.astype(float) / 255.0
            result_bgr = (enhanced_bgr * mask_3channel_float + \
                          image_to_preprocess_bgr * (1.0 - mask_3channel_float)).astype(np.uint8)
        else: # Fallback: simple CLAHE enhancement if no contours found for masking
            lab = cv2.cvtColor(image_to_preprocess_bgr, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l_enhanced = clahe.apply(l)
            enhanced_lab = cv2.merge((l_enhanced, a, b))
            result_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        return result_bgr

    def prepare_image_for_contour_detection(self, image_path_original, auto_preprocess=True):
        """
        Sets self.gray_for_contour, using the original image.
        The auto_preprocess logic has been removed.
        """
        if self.orig_image is None or self.loaded_orig_path != image_path_original:
             self.load_original_image(image_path_original)

        
        print("Using original image for contour detection (auto-preprocessing disabled).")
        image_to_consider_for_contour = self.orig_image.copy()
        
        
        self.image_for_contour = image_to_consider_for_contour
        self.gray_for_contour = cv2.cvtColor(self.image_for_contour, cv2.COLOR_BGR2GRAY)

    def find_document_contour(self, debug=False):
        if self.gray_for_contour is None:
            raise Exception("Grayscale image for contour detection not prepared. Call prepare_image_for_contour_detection first.")
        
        gray_input = self.gray_for_contour
        
        blurred = cv2.GaussianBlur(gray_input, (5, 5), 0) 
        edges = cv2.Canny(blurred, 50, 150)             
        
        kernel = np.ones((5, 5), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=5) 
        
        # Helper function for displaying resized images 
        def display_resized_image(window_name, image_to_show):
            img_h, img_w = image_to_show.shape[:2]
            scale_w, scale_h = 1.0, 1.0
            MAX_DISPLAY_WIDTH_DR = 1200 
            MAX_DISPLAY_HEIGHT_DR = 800
            if img_w > MAX_DISPLAY_WIDTH_DR: scale_w = MAX_DISPLAY_WIDTH_DR / img_w
            if img_h > MAX_DISPLAY_HEIGHT_DR: scale_h = MAX_DISPLAY_HEIGHT_DR / img_h
            display_scale_factor = min(scale_w, scale_h, 1.0)
            if display_scale_factor < 1.0:
                new_w, new_h = int(img_w * display_scale_factor), int(img_h * display_scale_factor)
                displayed_image = cv2.resize(image_to_show, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                displayed_image = image_to_show
            cv2.imshow(window_name, displayed_image)

        if debug:
            display_resized_image("DEBUG gray_input for find_contour", gray_input.copy())
            display_resized_image("Edges (for contour)", edges.copy()) 
            display_resized_image("Dilated Edges (for contour)", dilated_edges.copy()) 
            cv2.waitKey(0) 
            
            contours_for_debug_viz, _ = cv2.findContours(dilated_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours_for_debug_viz:
                contours_for_debug_viz = sorted(contours_for_debug_viz, key=cv2.contourArea, reverse=True)
                largest_contour_viz = contours_for_debug_viz[0]
                debug_contour_draw_img = cv2.cvtColor(dilated_edges.copy(), cv2.COLOR_GRAY2BGR)
                cv2.drawContours(debug_contour_draw_img, [largest_contour_viz], -1, (0,0,255), 2)
                actual_area_of_viz = cv2.contourArea(largest_contour_viz)
                print(f"DEBUG_CONTOUR_VIZ: Visualizing largest contour found by RETR_EXTERNAL. Its area: {actual_area_of_viz}")
                display_resized_image("DEBUG Largest Contour (Red) on Dilated Edges", debug_contour_draw_img)
                cv2.waitKey(0)
            else:
                print("DEBUG_CONTOUR_VIZ: No contours found to visualize for RETR_EXTERNAL check.")

        contours_found, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        height, width = gray_input.shape
        image_area = height * width
        min_area_ratio = 0.20
        required_area = min_area_ratio * image_area
        
        if debug:
            print(f"DEBUG_MAIN_LOGIC: Image Area: {image_area}, Required Contour Area for doc: {required_area}")
            if not contours_found:
                print("DEBUG_MAIN_LOGIC: cv2.findContours (main) returned no contours.")
            else:
                print(f"DEBUG_MAIN_LOGIC: Number of contours found (main): {len(contours_found)}")

        screenCnt = None
        if contours_found:
            contours_found = sorted(contours_found, key=cv2.contourArea, reverse=True)
            if debug and contours_found: # Check if list is not empty before accessing [0]
                 print(f"DEBUG_MAIN_LOGIC: Largest contour area (main): {cv2.contourArea(contours_found[0])}")

            for contour_item in contours_found[:5]:
                contour_area = cv2.contourArea(contour_item)
                if contour_area < required_area:
                    if debug: print(f"DEBUG_MAIN_LOGIC: Contour area {contour_area} is < required. Stopping poly approx.")
                    break 
                
                if debug: print(f"DEBUG_MAIN_LOGIC: Contour area {contour_area} is large enough. Approximating polygon...")
                peri = cv2.arcLength(contour_item, True)
                for epsilon_factor in [0.015, 0.02, 0.025, 0.03, 0.04, 0.05]:
                    epsilon = epsilon_factor * peri
                    approx = cv2.approxPolyDP(contour_item, epsilon, True)
                    if debug: print(f"DEBUG_MAIN_LOGIC:   Epsilon {epsilon_factor*100:.1f}%, Points: {len(approx)}")
                    if len(approx) == 4 and cv2.isContourConvex(approx):
                        if debug: print(f"DEBUG_MAIN_LOGIC:   Found 4-sided convex polygon!")
                        screenCnt = approx
                        break 
                if screenCnt is not None:
                    break
        
        if screenCnt is None:
            if debug: print("DEBUG_MAIN_LOGIC: No 4-sided convex polygon found from suitable large contours.")
            # Fallback logic: Check if contours_found is not empty before accessing contours_found[0]
            if contours_found and len(contours_found) > 0 and cv2.contourArea(contours_found[0]) > required_area:
                print("Could not find 4-sided convex contour, using minAreaRect of largest suitable contour.")
                rect = cv2.minAreaRect(contours_found[0])
                box = cv2.boxPoints(rect)
                screenCnt = np.int32(box)
            else:
                current_largest_area = 0
                if contours_found and len(contours_found) > 0 : current_largest_area = cv2.contourArea(contours_found[0])
                
                if contours_found and current_largest_area > 0 :
                    print(f"No suitable large contour found (largest area {current_largest_area} vs required {required_area}), defaulting to image margins.")
                else:
                     print("Critical error: No contours found at all or largest contour has zero area, defaulting to image margins.")
                margin = int(min(height, width) * 0.02)
                screenCnt = np.array([[[margin, margin]], [[width - margin, margin]], [[width - margin, height - margin]], [[margin, height - margin]]], dtype=np.int32)

        if debug:
            final_contour_display_img = self.image_for_contour.copy()
            if screenCnt is not None and screenCnt.size > 0 :
                 cv2.drawContours(final_contour_display_img, [screenCnt], -1, (0, 255, 0), 3)
            else:
                 if debug: print("DEBUG_MAIN_LOGIC: screenCnt is None or empty for final drawing.")
            display_resized_image("Detected Document Contour (on contour image)", final_contour_display_img)
            cv2.waitKey(0)
            
        return screenCnt

    def order_points(self, pts):
        pts = pts.reshape(4, 2)
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def four_point_transform(self, pts):
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect
        
        if self.orig_image is None:
            raise Exception("Original image not loaded for four_point_transform.")

        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
        if maxWidth <= 0 or maxHeight <= 0:
            print(f"Warning: Invalid dimensions for warp: maxWidth={maxWidth}, maxHeight={maxHeight}. Using default small size.")
            maxWidth = 100 # fallback
            maxHeight = 100 # fallback

        dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype=np.float32)
        
        M = cv2.getPerspectiveTransform(rect, dst)
        warped_output = cv2.warpPerspective(self.orig_image, M, (maxWidth, maxHeight))
        
        return warped_output

    def enhance_scanned_image(self, image, debug=False): # image is warped original
        """REVERTED: Enhancement for a cleaner, high-contrast look (CLAHE + Otsu)."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if debug: cv2.imshow("0. Grayscale for Enhance", gray); cv2.waitKey(0)

        # 1. CLAHE 
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced_gray = clahe.apply(gray)
        if debug: cv2.imshow("1. CLAHE Enhanced Gray", enhanced_gray); cv2.waitKey(0)

        # 2. Binarization using Otsu's method
        _, binary_image = cv2.threshold(enhanced_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if debug: cv2.imshow("2. Otsu Binarization", binary_image); cv2.waitKey(0)

        # 3. Ensure black text on white background
        if cv2.mean(binary_image)[0] < 128: 
            binary_image = cv2.bitwise_not(binary_image)
            if debug: cv2.imshow("3. Otsu Binarization (Inverted if needed)", binary_image); cv2.waitKey(0)
        
        # 4. Gentle Morphological Closing 
        kernel_clean = np.ones((2,2), np.uint8) 
        cleaned_binary = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel_clean, iterations=1)
        if debug: cv2.imshow("4. Cleaned Binary (After Gentle Close)", cleaned_binary); cv2.waitKey(0)

        result = cv2.cvtColor(cleaned_binary, cv2.COLOR_GRAY2BGR)
        
        if debug: cv2.imshow("Final Enhanced Result (CLAHE+Otsu Reverted)", result); cv2.waitKey(0)
        
        return result

    def scan_document(self, image_path_original, show_steps=False, save_result=True, enhance=True, auto_preprocess_contour=True):
        self.load_original_image(image_path_original)
        self.prepare_image_for_contour_detection(image_path_original, auto_preprocess=auto_preprocess_contour)
        document_contour = self.find_document_contour(debug=show_steps)
        
        if show_steps:
            contour_display_on_orig = self.orig_image.copy() 
            cv2.drawContours(contour_display_on_orig, [document_contour], -1, (0, 255, 0), 3)
            cv2.imshow("Detected Document (on original)", contour_display_on_orig)
            cv2.waitKey(0)
        
        warped = self.four_point_transform(document_contour)
        
        if show_steps:
            cv2.imshow("Warped Document (from original)", warped)
            cv2.waitKey(0)
        
        if enhance:
            result = self.enhance_scanned_image(warped, debug=show_steps)
        else:
            result = warped
        
        if save_result:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.output_dir}/scanned_{timestamp}.jpg"
            cv2.imwrite(filename, result)
            print(f"Scanned document saved as {filename}")
            
            if enhance: 
                warped_filename = f"{self.output_dir}/warped_orig_{timestamp}.jpg"
                cv2.imwrite(warped_filename, warped)
                print(f"Warped original document saved as {warped_filename}")
        
        if show_steps:
            cv2.destroyAllWindows()
            
        return result

    def interactive_scan(self, image_path):
        self.load_original_image(image_path) # Use load_original_image
        self.corners = []
        
        display_img_interactive = self.orig_image.copy()
        orig_h, orig_w = display_img_interactive.shape[:2]
        
        MAX_DISPLAY_WIDTH = 1200
        MAX_DISPLAY_HEIGHT = 800
        scale_w = 1.0
        scale_h = 1.0
        
        if orig_w > MAX_DISPLAY_WIDTH: scale_w = MAX_DISPLAY_WIDTH / orig_w
        if orig_h > MAX_DISPLAY_HEIGHT: scale_h = MAX_DISPLAY_HEIGHT / orig_h
            
        display_scale_factor = min(scale_w, scale_h, 1.0)
        
        if display_scale_factor < 1.0:
            new_w = int(orig_w * display_scale_factor)
            new_h = int(orig_h * display_scale_factor)
            display_img_interactive = cv2.resize(display_img_interactive, (new_w, new_h), interpolation=cv2.INTER_AREA)

        self.display_img_interactive_ref = display_img_interactive.copy() # Make a fresh copy for drawing
        
        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(self.corners) < 4:
                    self.corners.append([x, y])
                    cv2.circle(self.display_img_interactive_ref, (x, y), 5, (0, 255, 0), -1)
                    if len(self.corners) > 1:
                        cv2.line(self.display_img_interactive_ref, tuple(self.corners[-2]), tuple(self.corners[-1]), (0, 255, 0), 2)
                    if len(self.corners) == 4:
                        cv2.line(self.display_img_interactive_ref, tuple(self.corners[0]), tuple(self.corners[3]), (0, 255, 0), 2)
                    cv2.imshow("Select Document Corners", self.display_img_interactive_ref)
        
        print("Select the four corners of the document in order: Top-left, Top-right, Bottom-right, Bottom-left")
        print("Press 'r' to reset selection, 'c' to confirm (4 corners selected), ESC to cancel.")
        cv2.imshow("Select Document Corners", self.display_img_interactive_ref)
        cv2.setMouseCallback("Select Document Corners", click_event)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                self.corners = []
                self.display_img_interactive_ref = display_img_interactive.copy() 
                cv2.imshow("Select Document Corners", self.display_img_interactive_ref)
            elif key == ord('c') and len(self.corners) == 4:
                cv2.destroyAllWindows()
                break
            elif key == 27:
                return None
        
        scaled_corners = []
        if display_scale_factor < 1.0:
            for corner_point in self.corners: # Renamed to avoid clash
                scaled_x = int(corner_point[0] / display_scale_factor)
                scaled_y = int(corner_point[1] / display_scale_factor)
                scaled_corners.append([scaled_x, scaled_y])
        else:
            scaled_corners = self.corners
            
        corners_array = np.array(scaled_corners, dtype=np.float32)
        warped = self.four_point_transform(corners_array) 
        result = self.enhance_scanned_image(warped)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/scanned_{timestamp}.jpg"
        cv2.imwrite(filename, result)
        print(f"Scanned document saved as {filename}")
        
        warped_filename = f"{self.output_dir}/warped_orig_{timestamp}.jpg" # Save warped original
        cv2.imwrite(warped_filename, warped)
        
        final_display_h, final_display_w = result.shape[:2]
        if final_display_w > MAX_DISPLAY_WIDTH or final_display_h > MAX_DISPLAY_HEIGHT:
            final_scale_w = MAX_DISPLAY_WIDTH / final_display_w if final_display_w > MAX_DISPLAY_WIDTH else 1.0
            final_scale_h = MAX_DISPLAY_HEIGHT / final_display_h if final_display_h > MAX_DISPLAY_HEIGHT else 1.0
            final_display_scale = min(final_scale_w, final_scale_h, 1.0)
            result_display = cv2.resize(result, (0,0), fx=final_display_scale, fy=final_display_scale)
            cv2.imshow("Scanned Document", result_display)
        else:
            cv2.imshow("Scanned Document", result)
            
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return result

def main():
    parser = argparse.ArgumentParser(description="Document Scanner and Perspective Correction")
    parser.add_argument("image", help="Path to the image file")
    parser.add_argument("--interactive", "-i", action="store_true", 
                      help="Use interactive mode to select corners manually")
    parser.add_argument("--no-enhance", action="store_true", 
                      help="Skip image enhancement step")
    parser.add_argument("--show-steps", action="store_true", 
                      help="Show intermediate steps")
    parser.add_argument("--no-auto-preprocess", action="store_true",
                      help="Disable automatic preprocessing for contour detection")
    args = parser.parse_args()
    
    scanner = DocumentScanner()
    try:
        if args.interactive:
            scanner.interactive_scan(args.image)
        else:
            result = scanner.scan_document(
                args.image, 
                show_steps=args.show_steps, 
                enhance=not args.no_enhance,
                auto_preprocess_contour=not args.no_auto_preprocess
            )
            
            if result is not None and not args.show_steps:
                MAX_DISPLAY_WIDTH = 1200
                MAX_DISPLAY_HEIGHT = 800
                final_display_h, final_display_w = result.shape[:2]
                if final_display_w > MAX_DISPLAY_WIDTH or final_display_h > MAX_DISPLAY_HEIGHT:
                    final_scale_w = MAX_DISPLAY_WIDTH / final_display_w if final_display_w > MAX_DISPLAY_WIDTH else 1.0
                    final_scale_h = MAX_DISPLAY_HEIGHT / final_display_h if final_display_h > MAX_DISPLAY_HEIGHT else 1.0
                    final_display_scale = min(final_scale_w, final_scale_h, 1.0)
                    result_display = cv2.resize(result, (0,0), fx=final_display_scale, fy=final_display_scale)
                    cv2.imshow("Scanned Document", result_display)
                else:
                    cv2.imshow("Scanned Document", result)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()