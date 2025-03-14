# Smart Gate – Computer Vision & IoT  

##  Project Overview  
Smart Gate is an automated security system that combines **Computer Vision** and **IoT** to enhance access control. The system utilizes **face recognition** and **license plate recognition** to grant or deny entry, ensuring a **secure and efficient** authentication process.  

##  Features  
- **Face Recognition**: Implemented using **Haar Cascade** to verify authorized individuals.  
- **License Plate Recognition**: Uses **EasyOCR** to detect and read vehicle license plates.  
- **Arduino Uno Integration**: Controls the physical gate mechanism for **automated access**.  
- **Real-time Notifications**: Alerts are sent for **unauthorized access attempts**.  

##  Technologies Used  
- **Python** (for Computer Vision processing)  
- **OpenCV** (for Face Recognition using Haar Cascade)  
- **EasyOCR** (for License Plate Detection)  
- **Arduino Uno** (for gate control)  
- **Serial Communication** (for interaction between Python & Arduino)  

##  System Workflow  
1. **Face Recognition**: The system scans and verifies the user’s face.  
2. **License Plate Detection**: Captures and reads the vehicle’s license plate.  
3. **Access Decision**: If both face and plate are authorized, the gate opens.  
4. **Unauthorized Access Handling**: Sends real-time notifications upon detection of unregistered users.  

##  How It Works  
1. Run the Python script to start the system.  
2. The camera scans the user’s face and the vehicle’s license plate.  
3. If authorized, a command is sent to **Arduino Uno** to open the gate.  
4. If unauthorized, an alert notification is triggered.  

##  Future Improvements  
- **Integration with RFID for additional security**  
- **Cloud-based storage for access logs**  
- **AI-based anomaly detection for enhanced security**  

##  License  
This project is open-source and available under the **MIT License**.  

---

💡 *Smart Gate - A Step Towards Smarter and Safer Access Control!* 🚀  
