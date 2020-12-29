#!/usr/bin/env python
# encoding: utf-8
"""
@version: 1.0
@author: Tu Xinglong
@file: FinanceConst.py
@time: 2020/1/3 10:26
"""
"""
财务指标英文简称和中文名称对应关系
"""
# 原始财报数据
Choice_Finance_Report = {"BPS": "每股净资产",
                         "APTURNDAYS": "应付账款周转天数(含应付票据)",
                         "CASHTATIO": "现金比率",
                         "CATOASSET": "流动资产/总资产",  # (流动资产比率)
                         "CFPS": "每股现金流量净额",
                         "DEDUCTEDPROFITTOEBT": "扣除非经常性损益的净利润/净利润",
                         "EBITDAPS": "每股息税折旧摊销前利润",
                         "EPSBASIC": "每股收益EPS(基本)",
                         "INCOMESTATEMENT_14": "财务费用",
                         "INCOMESTATEMENT_13": "管理费用",
                         "GPMARGIN": "销售毛利率",
                         "INVTURNDAYS": "存货周转天数",
                         "BALANCESTATEMENT_94": "长期借款",
                         "INCOMESTATEMENT_60": "净利润",
                         "INCOMESTATEMENT_61": "归属于母公司股东的净利润",
                         "INCOMESTATEMENT_27": "营业支出",
                         "INCOMESTATEMENT_48": "营业利润",
                         "ORPS": "每股营业收入",
                         "ROA": "总资产报酬率ROA",
                         "ROEWA": "净资产收益率ROE(加权)",
                         "INCOMESTATEMENT_12": "销售费用",
                         "INCOMESTATEMENT_11": "税金及附加",
                         "INCOMESTATEMENT_83": "营业总收入",
                         "INCOMESTATEMENT_55": "利润总额",
                         "UNDISTRIBUTEDPS": "每股未分配利润",
                         "YOYEBT": "利润总额同比增长率",
                         "YOYEPSBASIC": "基本每股收益同比增长率",
                         "YOYNI": "净利润同比增长率",
                         "YOYROELILUTED": "净资产收益率同比增长率(摊薄)",
                         # =============以下为新增加因子=============
                         "INVTURNRATIO": "存货周转率",
                         "INCOMESTATEMENT_49": "营业外收入",
                         "BALANCESTATEMENT_25": "流动资产合计",
                         "BALANCESTATEMENT_93": "流动负债合计",
                         "BALANCESTATEMENT_17": "存货",  # 存货净额
                         "BALANCESTATEMENT_95": "应付债券",
                         "BALANCESTATEMENT_96": "长期应付款",
                         "EBIT": "息税前利润EBIT",
                         "EBITDA": "息税折旧摊销前利润",
                         "FCFF": "企业自由现金流量FCFF",
                         "ARTURNDAYS": "应收账款周转天数",
                         "ASSETTURNRATIO": "总资产周转率",
                         "INCOMESTATEMENT_9": "营业收入",
                         "INCOMESTATEMENT_10": "营业成本",
                         "DEDUCTEDINCOME": "归属于上市公司股东的扣除非经常性损益后的净利润",
                         "INCOMESTATEMENT_84": "营业总成本",
                         "INCOMESTATEMENT_20": "利息支出",
                         "INCOMESTATEMENT_11": "税金及附加",
                         "INCOMESTATEMENT_128": "利息收入",
                         "INCOMESTATEMENT_16": "公允价值变动收益",
                         "INCOMESTATEMENT_50": "营业外支出",
                         "BALANCESTATEMENT_31": "固定资产",
                         "BALANCESTATEMENT_39": "商誉",
                         "BALANCESTATEMENT_46": "非流动资产合计",
                         "BALANCESTATEMENT_103": "非流动负债合计",
                         "BALANCESTATEMENT_128": "负债合计",
                         "BALANCESTATEMENT_74": "资产总计",
                         "BALANCESTATEMENT_132": "未分配利润",
                         "BALANCESTATEMENT_131": "盈余公积",
                         "BALANCESTATEMENT_140": "归属于母公司股东权益合计",
                         "BALANCESTATEMENT_141": "股东权益合计",
                         "FCFFPS": "每股企业自由现金流量",
                         "NPMARGIN": "销售净利率",
                         "LIBILITYTOASSET": "资产负债率",
                         "ASSETSTOEQUITY": "权益乘数"}

# 日频因子数据
Choice_Finance_Factor = {"sestni_FY0": "归属母公司股东净利润",
                         "cagrpni_PY5": "过去五年营业总收入复合增长率",
                         "cagrgr_PY5": "过去五年归属母公司净利润复合增长率",
                         "eps_FY1": "一致预测每股收益(FY1)",
                         "sestni_FY1": "一致预测归属母公司净利润(FY1)",
                         "sestni_FY3": "一致预测归属母公司净利润(FY3)",
                         "sestni_YOY1": "未来一年一致预测净利润增长率",
                         "sestni_YOY3": "未来三年一致预测净利润增长率",
                         "": "",
                         "": "",
                         "": "",
                         "": "",
                         "": "",
                         "": "",
                         "": "",
                         "": "",}

# 日频因子数据
Tushare_Finance_Factor = {"open": "开盘价",
                          "close": "收盘价",
                          "high": "最高价",
                          "low": "最低价",
                          "open_qfq": "前复权开盘价",
                          "close_qfq": "前复权收盘价",
                          "high_qfq": "前复权最高价",
                          "low_qfq": "前复权最低价",
                          "open_hfq": "后复权开盘价",
                          "close_hfq": "后复权收盘价",
                          "amount": "成交量",
                          "volume": "成交额",
                          "ret": "个股收益率",
                          "turnover_rate": "换手率",
                          "pe_ttm": "市盈率TTM",
                          "total_mv": "总市值",
                          "total_ncl": "非流动负债合计",
                          "total_assets": "总资产",
                          "total_liab": "总负债",
                          "total_equity": "总权益",
                          "cfps": "每股自由现金流净额",
                          "volume_ratio": "量比",
                          "pb": "市净率",
                          "ps_ttm": "市销率(TTM)",
                          "dv_ttm": "股息率(TTM)",
                          "total_share": "总股本",
                          "float_share": "流通股本",
                          "free_share": "自由流通股本",
                          "circ_mv": "流通市值"}

# 日频因子数据
Wind_Finance_Factor = {
}




if __name__ == '__main__':
    pass