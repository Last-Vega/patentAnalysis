<template>
  <v-app>
    <Loading :flag="isShow" />
    <v-row no-gutters>
      <v-col cols="12" sm="9">
        <div v-if="this.updateCompany.length > 0 || this.updateTerm.length > 0">
            <div class="text-center">
                <v-btn color="red lighten-2" dark @click="updateZ">
                  更新する
                </v-btn>
            </div>
          </div>
        <ViewLatentSpace :options="options" :companyItems="CCContrib" :termItems="CTContrib" />
      </v-col>

      <v-col cols="12" sm="3">
        <ViewTabel :companyName="companyName" :termName="termName" :companyZ="this.options.series[0].data" :termZ="this.options.series[1].data" />
      </v-col>
    </v-row>
  </v-app>
</template>

<script>
import {
  companyTableData,
  termTableData,
  chartOptions,
  updateCompanyIndex,
  updateTermIndex
} from '@/components/createCompareLatentSpace.js'

import companyInfo from '@/assets/KajimaObayashiLatentComp0306.json'
import termInfo from '@/assets/KajimaObayashiLatentTerm0306.json'

import ViewTabel from '@/components/viewTable'
import ViewLatentSpace from '@/components/viewLatentInfo'
import Loading from '@/components/Loading'

export default {
  name: 'LatentSpace',
  components: {
    ViewTabel,
    ViewLatentSpace,
    Loading
    // Recommendation
  },
  data () {
    return {
      options: chartOptions,
      headers: [
        { text: 'Title', value: 'title' },
        { text: 'Authors', value: 'author' },
        { text: 'Conference', value: 'conference' },
        { text: 'Year', value: 'year' }
      ],
      companyItems: companyTableData,
      termItems: termTableData,
      companyName: [],
      companyXY: [],
      termName: [],
      termXY: [],
      query: '',
      updateCompany: updateCompanyIndex,
      updateTerm: updateTermIndex,
      isShow: false,
      maxCCPath: '',
      maxCTPath: '',
      CCContrib: '',
      CTContrib: ''
    }
  },
  methods: {
    makeScatter (company, term) {
      console.log(company)
      this.companyXY = company
      this.termXY = term
      this.options.series[0].data = company
      this.options.series[1].data = term
    },
    interpretation (ccPath, ctPath) {
      const path = {
        C: '企業',
        T: '技術用語',
        Y: '公開年',
        I: '筆頭IPC',
        F: 'Fターム',
        P: '-特許-'
      }
      const CCElmArray = ccPath.split('')
      const CTElmArray = ctPath.split('')

      for (const elm of CCElmArray) {
        console.log(elm)
        this.CCContrib += path[elm]
      }

      for (const elm of CTElmArray) {
        this.CTContrib += path[elm]
      }
      console.log(this.CCContrib)
    },
    async updateZ () {
      this.isShow = true
      console.log(this.updateComapny)
      console.log(this.updateTerm)
      const path = process.env.VUE_APP_BASE_URL + 'api/vsupdate'
      const postData = {
        companyZ: this.options.series[0].data,
        termZ: this.options.series[1].data,
        CompanyIndex: this.updateCompany,
        TermIndex: this.updateTerm
      }
      await this.$api
        .post(path, postData)
        .then(response => {
          this.updateCompany.splice(0, this.updateCompany.length)
          this.updateTerm.splice(0, this.updateTerm.length)
          console.log(response.data.company)
          this.makeScatter(response.data.company, response.data.term)
          this.isShow = false
          this.interpretation(response.data.maxCCPath, response.data.maxCTPath)
        })
        .catch(error => {
          console.log(error)
          this.isShow = false
        })
    }
  },
  created () {
    const companyData = companyInfo
    const termData = termInfo
    for (let i = 0; i < companyData.length; i++) {
      this.companyName.push(companyData[i].company)
      this.companyXY.push([companyData[i].x, companyData[i].y])
    }
    for (let i = 0; i < termData.length; i++) {
      this.termName.push(termData[i].term)
      this.termXY.push([termData[i].x, termData[i].y])
    }
    console.log(this.companyXY)
    this.options.series[0].dataLabal = this.companyName
    this.options.series[0].data = this.companyXY
    this.options.series[1].dataLabal = this.termName
    this.options.series[1].data = this.termXY
  }
}
</script>
