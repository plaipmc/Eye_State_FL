<h3 align="center">การจำแนกข้อมูลคลื่นไฟฟ้าสมองบนสถาปัตยกรรมแบบ Federated Learning</h3>
<h3 align="center">FEDERATED LEARNING FOR EEG-BASED CLASSIFICATION</h3>

#### ที่มาและความสำคัญ Introduction
โครงงานการจำแนกข้อมูลคลื่นกระแสไฟฟ้าบนสถาปัตยกรรมแบบการเรียนรู้แบบสหพันธ์นี้มีเป้าหมายเพื่อเรียนรู้การแปลงโครงสร้างสถาปัตยกรรมของโมเดลในการจำแนกข้อมูลคลื่นสมองแบบการเรียนรู้แบบรวมศูนย์ให้เป็นสถาปัตยกรรมแบบการเรียนรู้แบบสหพันธ์และผลกระทบต่อประสิทธิภาพของโมเดลเพื่อให้สามารถพัฒนาโมเดลเพื่อการจำแนกลักษณะบางอย่างที่รักษาความเป็นส่วนตัวของข้อมูลคลื่นสมองได้ เพื่อให้บรรลุเป้าหมายดังกล่าว จึงกำหนดวัตถุประสงค์ของโครงงานดังต่อไปนี้ 
1.	เพื่อศึกษาโมเดลการเรียนรู้แบบรวมศูนย์ที่เหมาะสมในการการจำแนกข้อมูลคลื่นสมองเพื่อนำมาใช้เป็นโมเดลในการเปรียบเทียบประสิทธิภาพ
2.	เพื่อศึกษาการแปลงโครงสร้างสถาปัตยกรรมของโมเดลในการจำแนกข้อมูลคลื่นสมองแบบการเรียนรู้แบบรวมศูนย์ให้เป็นโครงสร้างสถาปัตยกรรมแบบการเรียนรู้แบบสหพันธ์
3.	เพื่อทดสอบประสิทธิภาพของโมเดลการเรียนรู้แบบสหพันธ์ด้วยค่าความแม่นยำและค่า Recall พร้อมเปรียบเทียบกับโมเดลในรูปแบบการเรียนรู้แบบรวมศูนย์ 

#### Software ที่ใช้
* Ubuntu 22.04
* Python version 3.11.6
* torch version 2.1.0+cpu
* torchvision version 0.16.0+cpu
* Net version 1
* numpy version 1.24.3
* pandas version 2.1.2
* scikit-learn version 1.3.2
* NVFLARE version 2.4.0

#### ไฟล์และโฟลเดอร์ที่เกี่ยวข้อง
* dataset เป็นโฟลเดอร์ที่เก็บข้อมูล dataset ทั้งหมดที่ใช้และมีการแบ่งเป็น Training, Validating, Test dataset แล้ว
* preparedata เป็นโฟลเดอร์ที่เก็บไฟล์ที่ใช้ในการสร้างชุดข้อมูลให้แต่ละ client
* result เป็นโฟลเดอร์ที่รวบรวมผลลัพธ์ทั้งหมดจากการศึกษา
* ubuntu เป็นโฟลเดอร์ที่เก็บ job configuration folder และ dataset ที่เกี่ยวข้อง สามารถนำไปไว้ที่โปรแกรม Ubuntu เพื่อรันได้

#### ขั้นตอนการทำงาน
* ติดตั้ง  Ubuntu 22.04
* อัปเดตไลบรารี่
  ```sh
  sudo apt update
  sudo apt upgrade
  ```
* โคลน Git repo ของ NVFLARE มาติดตั้ง พร้อมใช้ version 2.4.0
  ```sh
  git clone https://github.com/NVIDIA/NVFlare.git
  cd NVFlare
  git checkout 2.4
  ```
* ติดตั้ง Library เพิ่มเติม
  ```sh
  sudo apt-get install python3-venv
  sudo apt install python3-pip
  python3 -m pip install --user --upgrade pip
  python3 -m pip install --user virtualenv
  ```
* ใช้งาน virtual environment สำหรับการรัน FL Simulation
  ```sh
  cd examples
  . set_env.sh
  ```
* ติดตั้ง Library เพิ่มเติมสำหรับ virtual environment นี้
  ```sh
  python3 -m pip install -U pip
  python3 -m pip install nvflare==2.4.0
  python3 -m pip install pandas
  python3 -m pip install scikit-learn
  ```
* นำโฟลเดอร์ทั้งหมดในโฟลเดอร์ 'ubuntu' ทั้งหมดไปไว้ใน Ubuntu 22.04
* ในการรัน FL simulation หนึ่งครั้งใช้คำสั่ง
  
  `cd <folder>`
  
  `nvflare simulator -n <number_of_clients> -t <number_of_thread> -gpu <gpu_name> ./jobs/<job_folder>/ -w <output_folder> | cat >> <filename>.txt`
  
  เช่น
  ```sh
  cd eyestatefixnn_fedprox_C5S5Com1
  nvflare simulator -n 5 -t 5 -gpu 0 ./jobs/eyestate_fedprox/ -w /tmp/nvflare/eyestate_fedprox_C5S5C1_lr5 | cat >> result_fedprox_C5S5C1_lr5_10local.txt
  ```

