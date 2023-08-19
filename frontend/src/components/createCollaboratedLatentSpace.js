import Highcharts from 'highcharts'
import More from 'highcharts/highcharts-more'
// import draggablePoints from 'highcharts/modules/draggable-points'

More(Highcharts)
// draggablePoints(Highcharts)
const companyTableData = {
  company: ''
}

const termTableData = {
  term: ''
}

const chartOptions = {
  chart: {
    width: Math.min(window.innerHeight, window.innerWidth) * 1.0,
    height: 100 + '%',
    zoomType: 'xy'
  },
  tooltip: {
    useHTML: true,
    formatter: function () {
      return this.point.label
    }
  },
  xAxis: {
    gridLineWidth: 1,
    minorTickInterval: 0.1,
    tickInterval: 0.2
  },
  yAxis: {
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
      animation: false
    },
    {
      name: '両方とも言及',
      data: [],
      dataLabal: [],
      type: 'scatter',
      animation: false
    },
    {
      name: '熊谷組のみ言及',
      data: [],
      dataLabal: [],
      type: 'scatter',
      animation: false
    },
    {
      name: '他者のみ言及',
      data: [],
      dataLabal: [],
      type: 'scatter',
      animation: false
    },
    {
      name: '両方言及なし',
      data: [],
      dataLabal: [],
      type: 'scatter',
      animation: false
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

export {
  companyTableData,
  termTableData,
  chartOptions
}
