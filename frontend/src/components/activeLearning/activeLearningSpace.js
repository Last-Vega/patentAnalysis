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
  tooltip: {
    useHTML: true,
    formatter: function () {
      // console.log(this.series.data.length)
      const flag = this.series.name
      const wrd = ''
      if (flag === 'Company') {
        const index = this.series.data.indexOf(this.point)
        const wrd = chartOptions.series[0].dataLabal[index]
        return wrd
      } else if (flag === 'Term') {
        const index = this.series.data.indexOf(this.point)
        const wrd = chartOptions.series[1].dataLabal[index]
        return wrd
      }
      return wrd
    }
  },
  xAxis: {
    min: -1,
    max: 1,
    gridLineWidth: 1,
    tickPixelInterval: 25
  },
  yAxis: {
    min: -1,
    max: 1,
    tickPixelInterval: 50
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
              // chartOptions.series[0].data[index] = [e.newPoint.x, e.newPoint.y]
              console.log({ x: e.newPoint.x, y: e.newPoint.y, company: e.target.company })
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
              // chartOptions.series[1].data[index] = [e.newPoint.x, e.newPoint.y]
              console.log(index)
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
// export { companyTableData, termTableData, chartOptions, updateCompanyIndex, updateTermIndex }
export { chartOptions }
