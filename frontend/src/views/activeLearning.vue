<template>
  <v-app>
    <!-- <Loading :flag="isShow" /> -->
    <v-container>
      <section class="charts">
        <highcharts :options="options" ref="chart"></highcharts>
      </section>
    </v-container>
    <v-btn id="add" depressed elevation="2" outlined v-on:click="addZero" color="primary">
      位置を決定する
    </v-btn>
  </v-app>
</template>

<script>
// import Loading from '@/components/Loading'
import data from '@/components/activeLearning/activeLearning.json'
import initData from '@/components/activeLearning/centrality_top_company.json'
// import {
//   companyTableData,
//   termTableData,
//   chartOptions,
//   updateCompanyIndex,
//   updateTermIndex
// } from '@/components/activeLearningSpace'
import {
  chartOptions
} from '@/components/activeLearning/activeLearningSpace'

export default {
  components: {
    // Loading
  },
  data () {
    return {
      // flag: false,
      options: chartOptions,
      // isShow: false,
      companyIndex: 0,
      termIndex: 0,
      queryData: data,
      queryIndex: 0,
      initalDataCompany: initData,
      companyName: [],
      companyXY: [],
      termName: [],
      termXY: []
    }
  },
  methods: {
    addZero () {
      // this.isShow = true
      if ('company' in this.queryData[this.queryIndex]) {
        const companyName = this.queryData[this.queryIndex].company
        this.options.series[0].dataLabal.push(companyName)
        this.options.series[0].data.push([0, 0])
        this.companyIndex += 1
        this.queryIndex += 1
      } else if ('term' in this.queryData[this.queryIndex]) {
        const termName = this.queryData[this.queryIndex].term
        console.log(termName)
        this.options.series[1].dataLabal.push(termName)
        this.options.series[1].data.push([0, 0])
        this.termIndex += 1
        this.queryIndex += 1
      }
      // this.isShow = false
    },
    // displayBibInfo () {
    //   return this.bibInfo.key[this.bibInfoIndex][0]
    // },
    async makeScatter () {
      // let nowBibInfoIndex = 0
      // const moved = []
      // const querySnapshot = await db.collection(this.collectionName).orderBy('ind', 'asc').get()
      // querySnapshot.forEach((postDoc) => {
      //   console.log(postDoc.data().ind, postDoc.data().title)
      //   moved.push([postDoc.data().x, postDoc.data().y])
      //   nowBibInfoIndex += 1
      // })
      // this.options.series[1].data = moved
      // this.bibInfoIndex = nowBibInfoIndex
    },
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
        this.termName.push(termData[i].term)
        this.termXY.push({
          dataIndex: i,
          x: termData[i].x,
          y: termData[i].y,
          label: termData[i].term,
          term: this.labelFormat(termData[i].term)
        })
      }
      // console.log(this.companyXY)
      this.options.series[0].dataLabal = this.companyName
      this.options.series[0].data = this.companyXY
      this.options.series[1].dataLabal = this.termName
      this.options.series[1].data = this.termXY
    },
    prep_init (companyData) {
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

      this.options.series[0].dataLabal = this.companyName
      this.options.series[0].data = this.companyXY
    }

  },
  created () {
    const companyData = this.initalDataCompany.key
    // const termData = []
    this.prep_init(companyData)
    if ('company' in this.queryData[this.queryIndex]) {
      this.options.series[0].data.push([0, 0])
      const companyName = this.queryData[this.queryIndex].company
      this.options.series[0].dataLabal.push(companyName)
      this.companyIndex += 1
      this.queryIndex += 1
    } else if ('term' in this.queryData[this.queryIndex]) {
      this.options.series[1].data.push([0, 0])
      const termName = this.queryData[this.queryIndex].term
      console.log(termName)
      this.options.series[1].dataLabal.push(termName)
      console.log(this.options.series[1].dataLabal)
      this.termIndex += 1
      this.queryIndex += 1
    }
  }
}
</script>
