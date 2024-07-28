<script lang="ts" setup>
import { ref, onMounted } from 'vue'
//调用ref,定义响应式数据
import { UploadFilled } from '@element-plus/icons-vue'
import type { UploadProps, UploadUserFile } from 'element-plus'
import { ElNotification } from 'element-plus'
import { reactive } from 'vue'

const input = ref('')
const fileName = ref('');
const plotData = ref('');
const plotoversample = ref('');
const variable = ref(['Temperature(℃)', 'Dissolved Oxygen (mg/L)', 'PH', 'Conductivity(μmhos/cm)', 'BOD (mg/L)', 'Nitrate N + Nitrite N(mg/L)',
  'Fecal Coliform (MPN/100ml)', 'Total Coliform (MPN/100ml)']);
const uploadUrl = ref('http://127.0.0.1:8000/waterquality/analyze');
const ADASYNUrl = ref('http://127.0.0.1:8000/waterquality/ADASYN');
const annUrl = ref('http://127.0.0.1:8000/waterquality/ANN');
const svmUrl = ref('http://127.0.0.1:8000/waterquality/SVM');
const KNNUrl = ref('http://127.0.0.1:8000/waterquality/KNN');
const gdbtUrl = ref('http://127.0.0.1:8000/waterquality/GDBT');
const modelUrl = ref('http://127.0.0.1:8000/waterquality/model_knn');
const smoteUrl = ref('http://127.0.0.1:8000/waterquality/smote');
const ada_row = ref()
const data_row = ref()
const model_results = ref('')
const model_scores = ref('')
const Accuracy_max = ref('')
const k_value = ref('')
const model = ref('')
const tableData = ref([])
const tableData1 = ref([])
const tableData2 = ref([])
const filepredict = ref()
const currentPage4 = ref(1)
const currentPage2 = ref(1)
const currentPage1 = ref(1)
const pageSize4 = ref(20)
const pageSize2 = ref(20)
const pageSize1 = ref(20)
const small = ref(false)
const background = ref(false)
const disabled = ref(false)
const total_number1 = ref(0)
const total_number2 = ref(0)
const total_number = ref(0)
const Currentdata = ref([])
const Currentdata1 = ref([])
const Currentdata2 = ref([])
const model_index = ref('模型指标')

const handleSizeChange = (val: number) => {
  pageSize4.value = val;
  const startIndex = (currentPage4.value - 1) * pageSize4.value;
  const endIndex = startIndex + pageSize4.value;
  Currentdata.value = tableData.value.slice(startIndex, endIndex)
  console.log(`${val} items per page`)
}

const handleCurrentChange = (val: number) => {
  currentPage4.value = val;
  const startIndex = (currentPage4.value - 1) * pageSize4.value;
  const endIndex = startIndex + pageSize4.value;
  Currentdata.value = tableData.value.slice(startIndex, endIndex)
  // 截取当前页的数据并返回
}

const handleSizeChange1 = (val: number) => {
  pageSize1.value = val;
  const startIndex = (currentPage1.value - 1) * pageSize1.value;
  const endIndex = startIndex + pageSize1.value;
  Currentdata1.value = tableData1.value.slice(startIndex, endIndex)
  console.log(`${val} items per page`)
}

const handleCurrentChange1 = (val: number) => {
  currentPage1.value = val;
  const startIndex = (currentPage1.value - 1) * pageSize1.value;
  const endIndex = startIndex + pageSize1.value;
  Currentdata1.value = tableData1.value.slice(startIndex, endIndex)
  // 截取当前页的数据并返回
}

const handleSizeChange2 = (val: number) => {
  pageSize2.value = val;
  const startIndex = (currentPage2.value - 1) * pageSize2.value;
  const endIndex = startIndex + pageSize2.value;
  Currentdata2.value = tableData2.value.slice(startIndex, endIndex)
  console.log(`${val} items per page`)
}

const handleCurrentChange2 = (val: number) => {
  currentPage2.value = val;
  const startIndex = (currentPage2.value - 1) * pageSize2.value;
  const endIndex = startIndex + pageSize2.value;
  Currentdata2.value = tableData2.value.slice(startIndex, endIndex)
  // 截取当前页的数据并返回
}


const img = ref([
  "1.png", "2.png", "3.png", "4.png"
])


//声明函数
const load_file = () => {
  ElNotification({
    title: 'Success',
    message: 'success to load file',
    type: 'success',
  })
}
const load_model = () => {
  ElNotification({
    title: 'Success',
    message: 'success to analyse',
    type: 'success',
  })
}
const load_ADA = () => {
  ElNotification({
    title: 'Success',
    message: 'success to complete oversample',
    type: 'success',
  })
}
//模型分析新数据集
const handlemodel = function (files) {
  const file = files.raw;
  filepredict.value = files.raw;
  // 创建一个 FormData 对象，用于将文件传输到后端
  const formData = new FormData();
  formData.append('file', file);
  fetch(modelUrl.value, {
    method: 'POST',
    body: formData
  })
    .then(response => response.json())
    .then(data => {
      // 在这里处理从后端返回的数据，例如显示数据分布图
      console.log('Received data from backend:', data);
      load_model();
      tableData.value = JSON.parse(data.value);
      const startIndex = (currentPage4.value - 1) * pageSize4.value;
      const endIndex = startIndex + pageSize4.value;
      Currentdata.value = tableData.value.slice(startIndex, endIndex)
      total_number.value = tableData.value.length;
    })
    .catch(error => console.error('Error:', error));
};

//最初上传文件分析
const handleFileUpload = function (files) {
  const file = files.raw;
  fileName.value = files.raw;
  // 创建一个 FormData 对象，用于将文件传输到后端
  const formData = new FormData();
  formData.append('file', file);

  fetch(uploadUrl.value, {
    method: 'POST',
    body: formData
  })
    .then(response => response.json())
    .then(data => {
      // 在这里处理从后端返回的数据，例如显示数据分布图
      console.log('Received data from backend:', data);
      load_file();
      plotData.value = `data:image/png;base64,${data.plot}`;
      data_row.value = data.row;
      tableData1.value = JSON.parse(data.value);
      const startIndex = (currentPage1.value - 1) * pageSize1.value;
      const endIndex = startIndex + pageSize1.value;
      Currentdata1.value = tableData1.value.slice(startIndex, endIndex)
      total_number1.value = tableData1.value.length;
    })
    .catch(error => console.error('Error:', error));
};

//进行过采样处理
const handleAdasyn = function () {
  // 创建一个 FormData 对象，用于将文件传输到后端
  const formData = new FormData();
  formData.append('file', fileName.value);
  fetch(ADASYNUrl.value, {
    method: 'POST',
    body: formData
  })
    .then(response => response.json())
    .then(data => {
      // 在这里处理从后端返回的数据，例如显示数据分布图
      console.log('Received data from backend:', data);
      load_ADA();
      plotoversample.value = `data:image/png;base64,${data.plot}`;
      ada_row.value = data.row
      tableData2.value = JSON.parse(data.value);
      const startIndex = (currentPage2.value - 1) * pageSize2.value;
      const endIndex = startIndex + pageSize2.value;
      Currentdata2.value = tableData2.value.slice(startIndex, endIndex)
      total_number2.value = tableData2.value.length;
    })
    .catch(error => console.error('Error:', error));
};


const handleSomte = function () {
  // 创建一个 FormData 对象，用于将文件传输到后端
  const formData = new FormData();
  formData.append('file', fileName.value);
  fetch(smoteUrl.value, {
    method: 'POST',
    body: formData
  })
    .then(response => response.json())
    .then(data => {
      // 在这里处理从后端返回的数据，例如显示数据分布图
      console.log('Received data from backend:', data);
      load_ADA();
      plotoversample.value = `data:image/png;base64,${data.plot}`;
      ada_row.value = data.row
      tableData2.value = JSON.parse(data.value);
      const startIndex = (currentPage2.value - 1) * pageSize2.value;
      const endIndex = startIndex + pageSize2.value;
      Currentdata2.value = tableData2.value.slice(startIndex, endIndex)
      total_number2.value = tableData2.value.length;
    })
    .catch(error => console.error('Error:', error));
};

//KNN分析，选取最合适的k值
const knn_request = function () {
  // 创建一个 FormData 对象，用于将文件传输到后端
  const formData = new FormData();
  formData.append('file', fileName.value);
  fetch(KNNUrl.value, {
    method: 'POST',
    body: formData
  })
    .then(response => response.json())
    .then(data => {
      // 在这里处理从后端返回的数据，例如显示数据分布图
      console.log('Received data from backend:', data);
      load_model();
      model_results.value = `data:image/png;base64,${data.plot}`;
      model_scores.value = `data:image/png;base64,${data.report_plot}`;
      Accuracy_max.value = data.row
      k_value.value = data.value
      model.value = 'KNN'
      model_index.value = '最优k值'
    })
    .catch(error => console.error('Error:', error));
}

const gdbt_request = function () {
  // 创建一个 FormData 对象，用于将文件传输到后端
  const formData = new FormData();
  formData.append('file', fileName.value);
  fetch(gdbtUrl.value, {
    method: 'POST',
    body: formData
  })
    .then(response => response.json())
    .then(data => {
      // 在这里处理从后端返回的数据，例如显示数据分布图
      console.log('Received data from backend:', data);
      load_model();
      model_results.value = `data:image/png;base64,${data.plot}`;
      model_scores.value = `data:image/png;base64,${data.report_plot}`;
      Accuracy_max.value = data.row
      k_value.value = data.value
      model.value = 'Gradient Boosting Decision Tree'
      model_index.value = '最优学习率'
      modelUrl.value = 'http://127.0.0.1:8000/waterquality/model_gdbt'
    })
    .catch(error => console.error('Error:', error));
}

const svm_request = function () {
  // 创建一个 FormData 对象，用于将文件传输到后端
  const formData = new FormData();
  formData.append('file', fileName.value);
  fetch(svmUrl.value, {
    method: 'POST',
    body: formData
  })
    .then(response => response.json())
    .then(data => {
      // 在这里处理从后端返回的数据，例如显示数据分布图
      console.log('Received data from backend:', data);
      load_model();
      model_results.value = `data:image/png;base64,${data.plot}`;
      model_scores.value = `data:image/png;base64,${data.report_plot}`;
      Accuracy_max.value = data.row
      k_value.value = data.value
      model.value = 'SVM'
      model_index.value = '最优c值'
      modelUrl.value = 'http://127.0.0.1:8000/waterquality/model_svm'
    })
    .catch(error => console.error('Error:', error));
}

const ann_request = function () {
  // 创建一个 FormData 对象，用于将文件传输到后端
  const formData = new FormData();
  formData.append('file', fileName.value);
  fetch(annUrl.value, {
    method: 'POST',
    body: formData
  })
    .then(response => response.json())
    .then(data => {
      // 在这里处理从后端返回的数据，例如显示数据分布图
      console.log('Received data from backend:', data);
      load_model();
      model_results.value = `data:image/png;base64,${data.plot}`;
      model_scores.value = `data:image/png;base64,${data.report_plot}`;
      Accuracy_max.value = data.row
      k_value.value = data.value
      model.value = 'ANN'
      model_index.value = '最优学习率'
      modelUrl.value = 'http://127.0.0.1:8000/waterquality/model_ann'
    })
    .catch(error => console.error('Error:', error));
}
</script>

<template>
  <el-scrollbar>
    <div class="common-layout" style="background-color: #e9e9eb;">
      <el-container style="background-color: white;width: 1700px; margin: auto">
        <el-header style="font-size: 50px;  height: 80px; text-align: center; margin-bottom: 30px;">
          water quality monitoring
        </el-header>
        <div class="img" style="height: 410px; width: 1600px; margin: auto;">
          <el-carousel :interval="4000" type="card" height="400px">
            <el-carousel-item v-for="(urls, i) in img" :key="i">
              <el-image :src="img[i]" fit="fill" style="height: 100%; width: 100%;" />
            </el-carousel-item>
          </el-carousel>
        </div>
        <el-divider>
            <span style="font-size: large; font-family: Arial, Helvetica, sans-serif;">变量解释</span>
          </el-divider>
          <div class="flex flex-wrap gap-4" style="display: flex; flex-direction: row; justify-content: center;" >
            <div>
              <el-card style="text-align: center; line-height: 25px; width: 830px;" shadow="hover">Temperature:<br />
              The optimum of freshwater bodies should approximatly be in the range 20°C to 30°C.
            </el-card>
            <el-card style="text-align: center; line-height: 25px; width: 830px;" shadow="hover">dissolved
              oxygen:<br />
              The value of dissolved oxygen should ideally be between 4 (mg/L) - 8 (mg/L).
            </el-card>
            <el-card style="text-align: center; line-height: 25px; width: 830px;" shadow="hover">PH:<br />
              PH value ranges from 6 - 8 for optimum water quality.
            </el-card>
            <el-card style="text-align: center; line-height: 25px; width: 830px;" shadow="hover">Conductivity:<br />
              Conductivityshould ideally be between 150 - 500 μmhos/cm to support diverse aquatic life.
            </el-card>
            </div>
            <div>
              <el-card style="text-align: center; line-height: 25px; width: 830px;" shadow="hover">BOD:<br />
              BOD has to be lower than 5 (mg/L) to obtain moderately clean water.
            </el-card>
            <el-card style="text-align: center; line-height: 25px; width: 830px;" shadow="hover">Nitrate and
              Nitrite:<br />
              The average of Nitrate and Nitrite should not exceed 5.5 (mg/L).
            </el-card>
            <el-card style="text-align: center; line-height: 25px; width: 830px;" shadow="hover">fecal coliform:<br />
              The value of fecal coliform should not exceed 200 MPN/100ml.
            </el-card>
            <el-card style="text-align: center; line-height: 25px; width: 830px;" shadow="hover">total coliform
              (TC):<br />
              The total coliform(TC) including faecal coliform should not exceed 500 MPN/100 ml.
            </el-card>
            </div>
            
          </div>

        <div style="display: flex; justify-content: center; flex-direction: column; margin: 20px;">
          <el-upload class="upload-demo" v-model="fileName" :action="uploadUrl" drag :on-change="handleFileUpload">
            <el-icon class="el-icon--upload"><upload-filled /></el-icon>
            <div class="el-upload__text">
              <em>click to upload</em>
            </div>
            <div class="el-upload__tip">
              jpg/png files with a size less than 500kb
            </div>
          </el-upload>
          <el-divider>
            <span style="font-size: large; font-family: Arial, Helvetica, sans-serif;">结果展示</span>
          </el-divider />
          <el-table :data="Currentdata1" style="width: 100%; margin-bottom: 20px;">
            <el-table-column v-for="(value, key) in Currentdata1[0]" :key="key" :prop="key"
              :label="key"></el-table-column>
          </el-table>
          <div class="demo-pagination-block" style="margin-left: 950px;">
            <el-pagination v-model:current-page="currentPage1" v-model:page-size="pageSize1"
              :page-sizes="[20, 30, 40, 50]" :small="small" :disabled="disabled" :background="background"
              layout="total, sizes, prev, pager, next, jumper" :total="total_number1" @size-change="handleSizeChange1"
              @current-change="handleCurrentChange1" />
          </div>
          <el-divider />
          <div style="margin: 20px auto;">
            <el-button type="primary" @click="handleAdasyn">ADASYN</el-button>
            <el-button type="primary" @click="handleSomte">SMOTE</el-button>
          </div>
        </div>
        <div style="display: flex; justify-content: center;">
          <el-empty :image="plotData" :image-size="450" description="原始数据分布"
            style="height: 500px; width: 550px; border-radius: 2px;" />
          <el-empty :image="plotoversample" :image-size="450" description="过采样后数据分布"
            style="height: 500px; width: 550px; border-radius: 2px" />
        </div>

        <el-divider />

        <el-row style="display: flex; justify-content: center;">
          <el-col :span="2" style="display: flex;justify-content: center;flex-direction: column;">
            <span>原始数据量</span>
            <span style="text-indent: 30px; line-height: 30px;">{{ data_row }}</span>
          </el-col>
          <el-col :span="2" style="display: flex;justify-content: center;flex-direction: column;">
            <span>重采样后数据</span>
            <span style="text-indent: 40px; line-height: 30px;">{{ ada_row }}</span>
          </el-col>
        </el-row>

        <el-divider>
          <span style="font-size: large; font-family: Arial, Helvetica, sans-serif;">结果展示</span>
        </el-divider />
        <el-table :data="Currentdata2" style="width: 100%; margin-bottom: 20px;">
          <el-table-column v-for="(value, key) in Currentdata2[0]" :key="key" :prop="key" :label="key"></el-table-column>
        </el-table>
        <div class="demo-pagination-block" style="margin-left: 950px;">
          <el-pagination v-model:current-page="currentPage2" v-model:page-size="pageSize2"
            :page-sizes="[20, 30, 40, 50]" :small="small" :disabled="disabled" :background="background"
            layout="total, sizes, prev, pager, next, jumper" :total="total_number2" @size-change="handleSizeChange2"
            @current-change="handleCurrentChange2" />
        </div>
        <el-divider />

        <div style="">
          <div style="display: flex; justify-content: center; margin: 20px;">
            <el-button type="primary" @click="knn_request">kNN</el-button>
            <el-button type="primary" @click="gdbt_request">GradientBoostingClassifier</el-button>
            <el-button type="primary" @click="svm_request">SVM</el-button>
            <el-button type="primary" @click="ann_request">ANN</el-button>
          </div>

          <el-divider />
          <div style="display: flex; justify-content: center;">
            <el-empty :image="model_results" :image-size="450" description="模型结果"
              style="height: 500px; width: 850px; border-radius: 2px;" />

            <el-upload ref="upload" class="upload-demo" action="" :limit="1" :on-change="handlemodel"
              style="display: flex; flex-direction: column; justify-content: center;">
              <el-button type="primary">模型分析</el-button>
            </el-upload>

            <el-empty :image="model_scores" :image-size="450" description="模型评分"
              style="height: 500px; width: 850px; border-radius: 2px" />
          </div>
          <el-divider />

          <el-row style="display: flex; justify-content: center;">
            <el-col :span="2" style="text-align: center;">
              <span style="display: block;">Accuracy_Max</span>
              <span style="line-height: 30px; display: block;">{{ Accuracy_max }}</span>
            </el-col>
            <el-col :span="2" style="text-align: center;">
              <span style="display: block;">{{ model_index }}</span>
              <span style="display: block; line-height: 30px;">{{ k_value }}</span>
            </el-col>
            <el-col :span="2" style="text-align: center;">
              <span style="display: block;">模型类型</span>
              <span style="line-height: 30px; display: block;">{{ model }}</span>
            </el-col>
          </el-row>

        </div>
      </el-container>
    </div>

    <div class="common-layout2" style="background-color: #e9e9eb;">
      <el-container style="background-color: white;width: 1700px; margin: auto">
        <el-main>
          <el-divider>
            <span style="font-size: large; font-family: Arial, Helvetica, sans-serif;">结果展示</span>
          </el-divider>

          <el-table :data="Currentdata" style="width: 100%; margin-bottom: 20px;">
            <el-table-column v-for="(value, key) in Currentdata[0]" :key="key" :prop="key"
              :label="key"></el-table-column>
          </el-table>
          <div class="demo-pagination-block" style="margin-left: 950px;">
            <el-pagination v-model:current-page="currentPage4" v-model:page-size="pageSize4"
              :page-sizes="[20, 30, 40, 50]" :small="small" :disabled="disabled" :background="background"
              layout="total, sizes, prev, pager, next, jumper" :total="total_number" @size-change="handleSizeChange"
              @current-change="handleCurrentChange" />
          </div>
          
        </el-main>
      </el-container>
    </div>
  </el-scrollbar>

</template>

<style scoped>
.box-card {
  display: flex;
  flex-direction: column;
  align-content: center;
  flex-wrap: wrap;
  height: 200px;
}


.demo-form-inline .el-input {
  --el-input-width: 220px;
}

.demo-form-inline .el-select {
  --el-select-width: 220px;
}

.citem {
  text-align: center;
}

.p {
  text-indent: 2px;
  font-size: 3px;
}


.demo-pagination-block+.demo-pagination-block {
  margin-top: 10px;
}

.demo-pagination-block .demonstration {
  margin-bottom: 16px;
}
</style>