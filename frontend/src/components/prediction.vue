<template>
  <v-app>
    <v-row no-gutters>
      <div>
        <div class="text-center top-right">
        </div>
        <highcharts :options="options" ref="chart" class="center"></highcharts>
      </div>
      <v-simple-table dense fixed-header height="800px">
          <template>
            <caption>
              単語
            </caption>
            <thead>
              <tr>
                <th class="text-left">name</th>
                <th class="text-left">属性</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="item in termAll" :key="item.dataIndex">
                <td>{{ item.label }}</td>
                <!-- <td class="text-right">
                  <v-text-field type="number" v-model.number="item.x"
                    @change="addUpdateIndex('company', item.dataIndex)" />
                </td>
                <td class="text-right">
                  <v-text-field type="number" v-model.number="item.y"
                    @change="addUpdateIndex('company', item.dataIndex)" />
                </td> -->
                <td v-if="item.color === 't1'" class="t1">
                  両方とも多く使用
                </td>
                <td v-else-if="item.color === 't2'" class="t2">
                  {{targetCompany}}のみ多く使用
                </td>
                <td v-else-if="item.color === 't3'" class="t3">
                  {{selectedCompany}}が多く使用
                </td>
                <td v-else-if="item.color === 't4'" class="t4">
                  両方ともあまり使用しない
                </td>
              </tr>
            </tbody>
          </template>
        </v-simple-table>
    </v-row>
  </v-app>
</template>

<script>
import {
  companyTableData,
  termTableData,
  chartOptions
} from '@/components/createCollaboratedLatentSpace'
// import ViewLatentSpace from '@/components/viewLatentInfo'

export default {
  name: 'LatentSpace',
  components: {
    // ViewLatentSpace
  },
  props: {
    responseData: {
      type: Object,
      required: true
    }
  },

  data () {
    return {
      options: chartOptions,
      isDraggable: true,
      companyItems: companyTableData,
      termItems: termTableData,
      companyName: [],
      companyXY: [],
      termName1: [],
      termName2: [],
      termName3: [],
      termName4: [],
      termXY1: [],
      termXY2: [],
      termXY3: [],
      termXY4: [],
      termAll: [],
      query: '',
      isShow: false,
      fromPointName: '',
      toPointName: '',
      history: [],
      targetCompany: '',
      selectedCompany: ''
    }
  },
  methods: {

    changeLabel (check) {
      const end = check ? 100 : 3
      this.companyXY.forEach((v, i) => {
        v.company = this.labelFormat(this.companyName[i], end)
      })
      this.termXY.forEach((v, i) => {
        v.term = this.labelFormat(this.termName[i], end)
      })
    },
    toggle () {
      this.isDraggable = !this.isDraggable
      this.options.series[0].dragDrop = {
        draggableX: this.isDraggable,
        draggableY: this.isDraggable,
        liveRedraw: this.isDraggable
      }
      this.options.series[1].dragDrop = {
        draggableX: this.isDraggable,
        draggableY: this.isDraggable,
        liveRedraw: this.isDraggable
      }
    },
    labelFormat (s, end = 3) {
      return s.replace('株式会社', '').slice(0, end)
    },
    makeScatter (company, term) {
      console.log(company)
      this.companyXY = company.map((v, i) => {
        return {
          dataIndex: i,
          label: this.companyName[i],
          x: v[0],
          y: v[1],
          company: this.labelFormat(this.companyName[i])
        }
      })
      this.termXY = term.map((v, i) => {
        return {
          dataIndex: i,
          label: this.termName[i],
          x: v[0],
          y: v[1],
          term: this.labelFormat(this.termName[i])
        }
      })
      this.options.series[0].data = this.companyXY
      this.options.series[1].data = this.termXY
    },
    prep (companyData, termData) {
      for (let i = 0; i < companyData.length; i++) {
        this.companyName.push(companyData[i].company)
        this.companyXY.push({
          dataIndex: i,
          x: companyData[i].x,
          y: companyData[i].y,
          label: companyData[i].company,
          company: this.labelFormat(companyData[i].company)
        })
      }
      for (let i = 0; i < termData.length; i++) {
        this.termAll.push(
          {
            dataIndex: i,
            x: termData[i].x,
            y: termData[i].y,
            label: termData[i].term,
            term: this.labelFormat(termData[i].term),
            color: termData[i].color
          }
        )
        if (termData[i].color === 't1') {
          console.log(termData[i])
          this.termXY1.push({
            dataIndex: i,
            x: termData[i].x,
            y: termData[i].y,
            label: termData[i].term,
            term: this.labelFormat(termData[i].term)
          })
        } else if (termData[i].color === 't2') {
          this.termXY2.push({
            dataIndex: i,
            x: termData[i].x,
            y: termData[i].y,
            label: termData[i].term,
            term: this.labelFormat(termData[i].term)
          })
        } else if (termData[i].color === 't3') {
          this.termXY3.push({
            dataIndex: i,
            x: termData[i].x,
            y: termData[i].y,
            label: termData[i].term,
            term: this.labelFormat(termData[i].term)
          })
        } else if (termData[i].color === 't4') {
          this.termXY4.push({
            dataIndex: i,
            x: termData[i].x,
            y: termData[i].y,
            label: termData[i].term,
            term: this.labelFormat(termData[i].term)
          })
        }
      }
      this.options.series[0].dataLabal = this.companyName
      this.options.series[0].data = this.companyXY

      this.options.series[1].dataLabal = this.termName1
      this.options.series[1].data = this.termXY1
      this.options.series[1].name = '両方とも多く使用'

      this.options.series[2].dataLabal = this.termName2
      this.options.series[2].data = this.termXY2
      this.options.series[2].name = this.targetCompany + 'のみ多く使用'

      this.options.series[3].dataLabal = this.termName3
      this.options.series[3].data = this.termXY3
      this.options.series[3].name = this.selectedCompany + 'のみ多く使用'

      this.options.series[4].dataLabal = this.termName4
      this.options.series[4].data = this.termXY4
      this.options.series[4].name = '両方ともあまり使用しない'
    }
  },
  async created () {
    const companyData = this.responseData.companyInfo
    const termData = this.responseData.termInfo
    this.targetCompany = this.responseData.targetCompany
    this.selectedCompany = this.responseData.selectedCompany
    console.log(companyData)
    console.log(termData)
    this.prep(companyData, termData)
  }
}
</script>

<style scoped>
.center {
  margin: 0 auto;
  width: 100%;
  height: 100%;
}
.top-right {
  position: absolute;
  top: 8px;
  right: 16px;
}
.t1 {
  background-color: #434348;
  color: #fff;
}

.t2 {
  background-color: #90ed7d;

}

.t3 {
  background-color: #f7a35c;
}

.t4 {
  background-color: #8085e9;
}
</style>
