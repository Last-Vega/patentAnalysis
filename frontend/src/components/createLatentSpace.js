import Highcharts from 'highcharts'
import More from 'highcharts/highcharts-more'
import draggablePoints from 'highcharts/modules/draggable-points'

More(Highcharts)
draggablePoints(Highcharts)
const companyTableData = {
  company: ''
}

const termTableData = {
  term: ''
}

const updateCompanyIndex = []
const updateTermIndex = []

const chartOptions = {
  chart: {
    width: Math.min(window.innerHeight, window.innerWidth) * 0.8,
    height: 100 + '%',
    zoomType: 'xy'
  },
  tooltip: {
    useHTML: true,
    formatter: function () {
      // console.log(this.series.data.length)
      const flag = this.series.data.length
      const wrd = ''
      if (flag <= 50) {
        const index = this.series.data.indexOf(this.point)
        const wrd = chartOptions.series[0].dataLabal[index]
        return wrd
      } else if (flag > 50) {
        const index = this.series.data.indexOf(this.point)
        const wrd = chartOptions.series[1].dataLabal[index]
        return wrd
      }
      return wrd
    }
  },
  xAxis: {
    // min: -1,
    // max: 1,
    gridLineWidth: 1,
    // tickPixelInterval: 25
    minorTickInterval: 0.1,
    tickInterval: 0.2
  },
  yAxis: {
    // min: -1,
    // max: 1,
    // tickPixelInterval: 50
    minorTickInterval: 0.1,
    tickInterval: 0.2
  },
  legend: {
    layout: 'vertical',
    align: 'left',
    verticalAlign: 'top',
    floating: true,
    backgroundColor: Highcharts.defaultOptions.chart.backgroundColor,
    borderWidth: 1
  },
  title: {
    text: '潜在空間'
  },
  series: [
    {
      name: 'Company',
      data: [],
      dataLabal: [],
      type: 'scatter',
      animation: false,
      dragDrop: {
        draggableX: true,
        draggableY: true,
        liveRedraw: true
      },
      point: {
        events: {
          mouseOver () {
            const point = this
            const index = point.index
            companyTableData.company = chartOptions.series[0].dataLabal[index]
          },
          drop: function (e) {
            const point = this
            const index = point.index
            if (e.newPoint.x !== undefined) {
              chartOptions.series[0].data[index] = { x: e.newPoint.x, y: e.newPoint.y, company: e.target.company }
            }
            updateCompanyIndex.push(index)
          }
        }
      }
    },
    {
      name: 'Term',
      data: [],
      dataLabal: [],
      type: 'scatter',
      color: 'red',
      animation: false,
      dragDrop: {
        draggableX: true,
        draggableY: true,
        liveRedraw: true
      },
      point: {
        events: {
          mouseOver () {
            const point = this
            const index = point.index
            termTableData.term = chartOptions.series[1].dataLabal[index]
          },
          drop: function (e) {
            const point = this
            const index = point.index
            if (e.newPoint.x !== undefined) {
              chartOptions.series[1].data[index] = { x: e.newPoint.x, y: e.newPoint.y, term: e.target.term }
            }
            updateTermIndex.push(index)
          }
        }
      }
    }
  ],
  plotOptions: {
    series: {
      states: {
        hover: {
          enabled: false
        }
      },
      dataLabels: {
        enabled: true,
        allowOverlap: true,
        format: '{point.company}{point.term}'
      }
    }
  }
}

export { companyTableData, termTableData, chartOptions, updateCompanyIndex, updateTermIndex }
