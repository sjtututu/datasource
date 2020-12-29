#!/usr/bin/env python
# encoding: utf-8
"""
@version: 1.0
@author: Tu Xinglong
@file: test.py
@time: 2019/12/25 10:03
"""
"""
test the financal data from choice!
"""
__author__ = "Tu Xinglong"
__version__ = "1.0.1"

# 标准库
import os
import sys

# 第三方库
import pandas as pd
import numpy as np
import tushare as ts
from dateutil.parser import parse
from datetime import datetime
import warnings
from EmQuantAPI import *  # noqa

c.start()  # noqa

CurrentPath = os.path.abspath(os.path.dirname(__file__))  # 设置绝对路径
Pre_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path += [CurrentPath, Pre_path, Pre_path + '\\Engine']
data_path = Pre_path + '\\FactorData\\FinanceReport\\'


# def finance_report():
#     """
#     下载所有股票原始财务报表
#     :return:
#     """
#     data = c.sector("001004", "2019-12-26")
#     stock_list = [stock_code for i, stock_code in enumerate(data.Data) if i % 2 == 0]
#     report_list =["20090930","20091231",
#                   "20100331","20100630","20100930","20101231","20110331","20110630","20110930","20111231",
#                   "20120331","20120630","20120930","20121231","20130331","20130630","20130930","20131231",
#                   "20140331","20140630","20140930","20141231","20150331","20150630","20150930","20151231",
#                   "20160331","20160630","20160930","20161231","20170331","20170630","20170930","20171231",
#                   "20180331","20180630","20180930","20181231","20190331","20190630","20190930"]#,"20191231"
#     for stock_code in stock_list:
#         # 遍历所有股票
#         #TAXPAY,OTHERCINCOME,NETPROFIT重复的指标代码
#         for report_date in report_list:
#             # 资产负债表
#             BalanceStatement = c.ctr("BalanceStatementSHSZ",
#                                      "REPORTDATE,STR_ISLISTINGREPORT,reportType ,STR_COMBINETYPE,MONETARYFUND,"
#                                      "SETTLEMENTPROVISION,LENDFUND,TRADE_FINASSET_NOTFVTPL,MARGINOUTFUND,FVALUEFASSET,"
#                                      "TRADEFASSET,DERIVEFASSET,ACCOUNTBILLREC,BILLREC,ACCOUNTREC,FINANCE_RECE,ADVANCEPAY,"
#                                      "PREMIUMREC,RIREC,RICONTACTRESERVEREC,TOTAL_OTHER_RECE,INTERESTREC,EXPORTREBATEREC,"
#                                      "SUBSIDYREC,INTERNALREC,BUYSELLBACKFASSET,AMORCOSTFASSET,FVALUECOMPFASSET,INVENTORY,"
#                                      "CONTRACTASSET,CLHELDSALEASS,NONLASSETONEYEAR,DLYWZC,OTHERLASSET,LASSETOTHER,LASSETBALANCE,"
#                                      "SUMLASSET,LOANADVANCES,CREDINV,AMORCOSTFASSETFLD,OTHCREDINV,FVALUECOMPFASSETFLD,"
#                                      "SALEABLEFASSET,HELDMATURITYINV,LTREC,LTEQUITYINV,OTHEREQUITYINV,OTHERNONFASSET,ESTATEINVEST,"
#                                      "FIXEDASSET,CONSTRUCTIONPROGRESS,CONSTRUCTIONMATERIAL,LIQUIDATEFIXEDASSET,PRODUCTBIOLOGYASSET,"
#                                      "OILGASASSET,USERIGHT_ASSET,INTANGIBLEASSET,DEVELOPEXP,GOODWILL,LTDEFERASSET,DEFERINCOMETAXASSET,"
#                                      "OTHERNONLASSET,NONLASSETOTHER,NONLASSETBALANCE,SUMNONLASSET,ASSETOTHER,ASSETBALANCE,SUMASSET,"
#                                      "STBORROW,BORROWFROMCBANK,DEPOSIT,BORROWFUND,TRADE_FINLIAB_NOTFVTPL,FVALUEFLIAB,TRADEFLIAB,"
#                                      "DERIVEFLIAB,ACCOUNTBILLPAY,BILLPAY,ACCOUNTPAY,ADVANCERECEIVE,CONTRACTLIAB,SELLBUYBACKFASSET,"
#                                      "COMMPAY,SALARYPAY,TAXPAY,TOTAL_OTHER_PAYABLE,INTERESTPAY,RIPAY,INTERNALPAY,ANTICIPATELLIAB,"
#                                      "CONTACTRESERVE,AGENTTRADESECURITY,AGENTUWSECURITY,DEFERINCOMEONEYEAR,AMORCOSTFLIAB,STBONDREC,"
#                                      "CLHELDSALELIAB,NONLLIABONEYEAR,DLYWFZ,OTHERLLIAB,LLIABOTHER,LLIABBALANCE,SUMLLIAB,LTBORROW,"
#                                      "AMORCOSTFLIABFLD,BONDPAY,LEASE_LIAB,LTACCOUNTPAY,LTSALARYPAY,SPECIALPAY,ANTICIPATELIAB,"
#                                      "DEFERINCOME,DEFERINCOMETAXLIAB,OTHERNONLLIAB,NONLLIABOTHER,NONLLIABBALANCE,SUMNONLLIAB,"
#                                      "LIABOTHER,LIABBALANCE,SUMLIAB,SHARECAPITAL,QTQYGJ,CAPITALRESERVE,INVENTORYSHARE,OTHERCINCOME,"
#                                      "SPECIALRESERVE,SURPLUSRESERVE,GENERALRISKPREPARE,UNCONFIRMINVLOSS,RETAINEDEARNING,PLANCASHDIVI,"
#                                      "DIFFCONVERSIONFC,PARENTEQUITYOTHER,PARENTEQUITYBALANCE,SUMPARENTEQUITY,MINORITYEQUITY,SHEQUITYOTHER,"
#                                      "SHEQUITYBALANCE,SUMSHEQUITY,LIABSHEQUITYOTHER,LIABSHEQUITYBALANCE,SUMLIABSHEQUITY,",
#                                      "secucode=" + stock_code + ", ReportDate=" + report_date + ", ReportType=1")
#             if BalanceStatement.ErrorCode == 0:
#                 BalanceStatement = pd.DataFrame(BalanceStatement.Data, BalanceStatement.Indicators).T
#                 BalanceStatement.index = BalanceStatement.REPORTDATE
#                 # 现金流量表
#                 # 该表主要提供了证券代码为沪深股票品种，公司类型为通用的现金流量表 参数: 证券代码 报告期 报表类型 字段: 报告日期 上市前/上市后 报表类型 公司类型 销售商品、提供劳务收到的现金(元) 客户存款和同业存放款项净增加额(元) 向中央银行借款净增加额(元) 向其他金融机构拆入资金净增加额(元) 收到原保险合同保费取得的现金(元) 收到再保险业务现金净额(元) 保户储金及投资款净增加额(元) 处置交易性金融资产净增加额(元) 收取利息、手续费及佣金的现金(元) 拆入资金净增加额(元) 发放贷款及垫款的净减少额(元) 回购业务资金净增加额(元) 收到的税费返还(元) 收到其他与经营活动有关的现金(元) 经营活动现金流入的其他项目(元) 经营活动现金流入的平衡项目(元) 经营活动现金流入小计(元) 购买商品、接受劳务支付的现金(元) 客户贷款及垫款净增加额(元) 存放中央银行和同业款项净增加额(元) 支付原保险合同赔付款项的现金(元) 支付利息、手续费及佣金的现金(元) 支付保单红利的现金(元) 支付给职工以及为职工支付的现金(元) 支付的各项税费(元) 支付其他与经营活动有关的现金(元) 经营活动现金流出的其他项目(元) 经营活动现金流出的平衡项目(元) 经营活动现金流出小计(元) 经营活动产生的现金流量净额其他项目(元) 经营活动产生的现金流量净额平衡项目(元) 经营活动产生的现金流量净额(元) 收回投资收到的现金(元) 取得投资收益收到的现金(元) 处置固定资产、无形资产和其他长期资产收回的现金净额(元) 处置子公司及其他营业单位收到的现金净额(元) 减少质押和定期存款所收到的现金(元) 收到其他与投资活动有关的现金(元) 投资活动现金流入的其他项目(元) 投资活动现金流入的平衡项目(元) 投资活动现金流入小计(元) 购建固定资产、无形资产和其他长期资产支付的现金(元) 投资支付的现金(元) 质押贷款净增加额(元) 取得子公司及其他营业单位支付的现金净额(元) 增加质押和定期存款所支付的现金(元) 支付其他与投资活动有关的现金(元) 投资活动现金流出的其他项目(元) 投资活动现金流出的平衡项目(元) 投资活动现金流出小计(元) 投资活动产生的现金流量净额其他项目(元) 投资活动产生的现金流量净额平衡项目(元) 投资活动产生的现金流量净额(元) 吸收投资收到的现金(元) 其中:子公司吸收少数股东投资收到的现金(元) 取得借款收到的现金(元) 发行债券收到的现金(元) 收到其他与筹资活动有关的现金(元) 筹资活动现金流入的其他项目(元) 筹资活动现金流入的平衡项目(元) 筹资活动现金流入小计(元) 偿还债务支付的现金(元) 分配股利、利润或偿付利息支付的现金(元) 其中:子公司支付给少数股东的股利、利润(元) 购买子公司少数股权而支付的现金(元) 支付其他与筹资活动有关的现金(元) 其中:子公司减资支付给少数股东的现金(元) 筹资活动现金流出的其他项目(元) 筹资活动现金流出的平衡项目(元) 筹资活动现金流出小计(元) 筹资活动产生的现金流量净额其他项目(元) 筹资活动产生的现金流量净额平衡项目(元) 筹资活动产生的现金流量净额(元) 汇率变动对现金及现金等价物的影响(元) 现金及现金等价物净增加额其他项目(元) 现金及现金等价物净增加额平衡项目(元) 现金及现金等价物净增加额(元) 加:期初现金及现金等价物余额(元) 期末现金及现金等价物余额其他项目(元) 期末现金及现金等价物余额平衡项目(元) 期末现金及现金等价物余额(元) 净利润(元) 资产减值准备(元) 固定资产和投资性房地产折旧(元) 其中:固定资产折旧、油气资产折耗、生产性生物资产折旧(元) 投资性房地产折旧(元) 无形资产摊销(元) 长期待摊费用摊销(元) 递延收益摊销(元) 待摊费用的减少(元) 预提费用的增加(元) 处置固定资产、无形资产和其他长期资产的损失(元) 固定资产报废损失(元) 公允价值变动损失(元) 财务费用(元) 投资损失(元) 递延所得税(元) 其中:递延所得税资产减少(元) 递延所得税负债增加(元) 预计负债的增加(元) 存货的减少(元) 经营性应收项目的减少(元) 经营性应付项目的增加(元) 其他(元) 经营活动产生的现金流量净额其他项目(元) 经营活动产生的现金流量净额平衡项目(元) 经营活动产生的现金流量净额(元) 债务转为资本(元) 一年内到期的可转换公司债券(元) 融资租入固定资产(元) 不涉及现金收支的投资和筹资活动金额其他项目(元) 现金的期末余额(元) 减:现金的期初余额(元) 加:现金等价物的期末余额(元) 减:现金等价物的期初余额(元) 现金及现金等价物净增加额其他项目(元) 现金及现金等价物净增加额平衡项目(元) 现金及现金等价物的净增加额(元) 公告日期 数据来源 审计意见(境内)
#                 CashFlowStatement = c.ctr("CashFlowStatementSHSZ", "SALEGOODSSERVICEREC,NIDEPOSIT,"
#                                           "NIBORROWFROMCBANK,NIBORROWFROMFI,PREMIUMREC,NETRIREC,NIINSUREDDEPOSITINV,NIDISPTRADEFASSET,"
#                                           "INTANDCOMMREC,NIBORROWFUND,NDLOANADVANCES,NIBUYBACKFUND,TAXRETURNREC,OTHEROPERATEREC,"
#                                           "OPERATEFLOWINOTHER,OPERATEFLOWINBALANCE,SUMOPERATEFLOWIN,BUYGOODSSERVICEPAY,NILOANADVANCES,"
#                                           "NIDEPOSITINCBANKFI,INDEMNITYPAY,INTANDCOMMPAY,DIVIPAY,EMPLOYEEPAY,OTHEROPERATEPAY,"
#                                           "OPERATEFLOWOUTOTHER,OPERATEFLOWOUTBALANCE,SUMOPERATEFLOWOUT,OPERATEFLOWOTHER,OPERATEFLOWBALANCE,"
#                                           "NETOPERATECASHFLOW,DISPOSALINVREC,INVINCOMEREC,DISPFILASSETREC,DISPSUBSIDIARYREC,REDUCEPLEDGETDEPOSIT,"
#                                           "OTHERINVREC,INVFLOWINOTHER,INVFLOWINBALANCE,SUMINVFLOWIN,BUYFILASSETPAY,INVPAY,NIPLEDGELOAN,"
#                                           "GETSUBSIDIARYPAY,ADDPLEDGETDEPOSIT,OTHERINVPAY,INVFLOWOUTOTHER,INVFLOWOUTBALANCE,SUMINVFLOWOUT,"
#                                           "INVFLOWOTHER,INVFLOWBALANCE,NETINVCASHFLOW,ACCEPTINVREC,SUBSIDIARYACCEPT,LOANREC,ISSUEBONDREC,"
#                                           "OTHERFINAREC,FINAFLOWINOTHER,FINAFLOWINBALANCE,SUMFINAFLOWIN,REPAYDEBTPAY,DIVIPROFITORINTPAY,"
#                                           "SUBSIDIARYPAY,BUYSUBSIDIARYPAY,OTHERFINAPAY,SUBSIDIARYREDUCTCAPITAL,FINAFLOWOUTOTHER,FINAFLOWOUTBALANCE,"
#                                           "SUMFINAFLOWOUT,FINAFLOWOTHER,FINAFLOWBALANCE,NETFINACASHFLOW,EFFECTEXCHANGERATE,NICASHEQUIOTHER,"
#                                           "NICASHEQUIBALANCE,NICASHEQUI,CASHEQUIBEGINNING,CASHEQUIENDINGOTHER,CASHEQUIENDINGBALANCE,CASHEQUIENDING,"
#                                           "NETPROFIT,ASSETDEVALUE,FIXANDESTATEDEPR,FIXEDASSETETCDEPR,ESTATEINVESTDEPR,INTANGIBLEASSETAMOR,"
#                                           "LTDEFEREXPAMOR,DEFERINCOMEAMOR,DEFEREXPREDUCE,DRAWINGEXPADD,DISPFILASSETLOSS,FIXEDASSETLOSS,FVALUELOSS,"
#                                           "FINANCEEXP,INVLOSS,DEFERTAX,DEFERTAXASSETREDUCE,DEFERTAXLIABADD,ANTICIPATELIABADD,INVENTORYREDUCE,"
#                                           "OPERATERECREDUCE,OPERATEPAYADD,OTHER,DEC_JYHDCSDXJLLJEQT,DEC_JYHDCSDXJLLJEPH,DEC_JYHDCSDXJLLJE,"
#                                           "DEBTTOCAPITAL,CBONEYEAR,FINALEASEFIXEDASSET,NOREFERCASHOTHER,DEC_XJDQMYE,DEC_XJDQCYE,DEC_XJDJWQMYE,"
#                                           "DEC_XJDJWQCYE,DEC_XJJZJECETS,DEC_XJJZJECEHJ,DEC_XJJXJDJWJZJ",
#                                           "secucode=" + stock_code + ", ReportDate=" + report_date + ", ReportType=1")
#                 CashFlowStatement = pd.DataFrame(CashFlowStatement.Data, CashFlowStatement.Indicators).T
#                 CashFlowStatement.index = BalanceStatement.REPORTDATE
#                 # 利润表
#                 # 该表主要提供了证券代码为沪深股票品种，公司类型为通用的利润表 参数: 证券代码 报告期 报表类型 字段: 报告日期 上市前/上市后 报表类型 公司类型 一、营业总收入(元) 营业收入(元) 利息收入(元) 已赚保费(元) 手续费及佣金收入(元) 其他业务收入(元) 营业总收入其他项目(元) 二、营业总成本(元) 营业成本(元) 利息支出(元) 手续费及佣金支出(元) 研发费用(元) 退保金(元) 赔付支出净额(元) 提取保险合同准备金净额(元) 保单红利支出(元) 分保费用(元) 其他业务成本(元) 营业税金及附加(元) 销售费用(元) 管理费用(元) 财务费用(元) 其中:利息费用(元) 其中:利息收入(元) 资产减值损失(元) 信用减值损失(元) 营业总成本其他项目(元) 加:公允价值变动收益(元) 投资收益(元) 其中:对联营企业和合营企业的投资收益(元) 净敞口套期收益(元) 资产处置收益(元) 汇兑收益(元) 其他收益(元) 营业利润其他项目(元) 营业利润平衡项目(元) 三、营业利润(元) 加:营业外收入(元) 其中:非流动资产处置利得(元) 减:营业外支出(元) 其中:非流动资产处置净损失(元) 加:影响利润总额的其他项目(元) 利润总额平衡项目(元) 四、利润总额(元) 减:所得税费用(元) 加:影响净利润的其他项目(元) 未确认投资损失(元) 净利润差额(合计平衡项目)(元) 五、净利润(元) 其中:被合并方在合并前实现利润(元) 持续经营净利润(元) 终止经营净利润(元) 归属于母公司股东的净利润(元) 少数股东损益(元) 净利润其他项目(元) 净利润差额(合计平衡项目)(元) (一)基本每股收益(元) (二)稀释每股收益(元) 七、其他综合收益(元) 归属于母公司股东的其他综合收益(元) 归属于少数股东的其他综合收益(元) 八、综合收益总额(元) 归属于母公司所有者的综合收益总额(元) 归属于少数股东的综合收益总额(元) 公告日期 数据来源 审计意见(境内)
#                 IncomeStatement = c.ctr("IncomeStatementSHSZ", "TOTALOPERATEREVE,OPERATEREVE,"
#                                         "INTREVE,PREMIUMEARNED,COMMREVE,OTHERREVE,TOTALOPERATEREVEOTHER,TOTALOPERATEEXP,OPERATEEXP,"
#                                         "INTEXP,COMMEXP,RDEXP,SURRENDERPREMIUM,NETINDEMNITYEXP,NETCONTACTRESERVE,POLICYDIVIEXP,RIEXP,"
#                                         "OTHEREXP,OPERATETAX,SALEEXP,MANAGEEXP,FINANCEEXP,QZLXFY,QZLXSR,ASSETDEVALUELOSS,XYZJSZ,"
#                                         "TOTALOPERATEEXPOTHER,FVALUEINCOME,INVESTINCOME,INVESTJOINTINCOME,JCKTQSY,ADISPOSALINCOME,"
#                                         "EXCHANGEINCOME,MIOTHERINCOME,OPERATEPROFITOTHER,OPERATEPROFITBALANCE,OPERATEPROFIT,NONOPERATEREVE,"
#                                         "FLDZCCZLD,NONOPERATEEXP,NONLASSETNETLOSS,SUMPROFITOTHER,SUMPROFITBALANCE,SUMPROFIT,INCOMETAX,"
#                                         "NETPROFITOTHER1,UNCONFIRMINVLOSS,NETPROFITBALANCE2,COMBINEDNETPROFITB,CONTINUOUSONPROFIT,"
#                                         "TERMINATIONONPROFIT,PARENTNETPROFIT,MINORITYINCOME,NETPROFITOTHER2,NETPROFITBALANCE1,BASICEPS,"
#                                         "DILUTEDEPS,PARENTOTHERCINCOME,MINORITYOTHERCINCOME,SUMCINCOME,PARENTCINCOME,MINORITYCINCOME,"
#                                         "FIRSTNOTICEDATE,REPORTSOURCETYPE,OPINIONTYPE",
#                                         "secucode=" + stock_code + ", ReportDate=" + report_date + ", ReportType=1")
#                 IncomeStatement = pd.DataFrame(IncomeStatement.Data, IncomeStatement.Indicators).T
#                 IncomeStatement.index = BalanceStatement.REPORTDATE
#                 # 合并三大报表
#                 finance_data = pd.concat([pd.concat([BalanceStatement, CashFlowStatement], axis=1), IncomeStatement],
#                                          axis=1)
#                 if not os.path.isdir(data_path + stock_code[0:6]):
#                     os.mkdir(data_path + stock_code[0:6])
#                     finance_data.to_hdf(data_path + stock_code[0:6] + "\\" + report_date + ".h5", report_date)
#                 else:
#                     finance_data.to_hdf(data_path + stock_code[0:6] + "\\" + report_date + ".h5", report_date)
#             else:
#                 continue
#
#     return

def finance_report():
    # 沪深股票 每股净资产BPS 应付账款周转天数(含应付票据) 现金比率 流动资产/总资产 每股现金流量净额 扣除非经常性损益的净利润/净利润
    # 每股息税折旧摊销前利润 每股收益EPS(基本) 财务费用 管理费用 销售毛利率 存货周转天数 长期借款 净利润 归属于母公司股东的净利润
    # 营业支出 营业利润 每股营业收入 总资产报酬率ROA 净资产收益率ROE(加权) 销售费用 税金及附加 营业总收入 利润总额 每股未分配利润
    # 利润总额同比增长率 基本每股收益同比增长率 净利润同比增长率 净资产收益率同比增长率(摊薄)
    report_list = ["20090331", "20090630", "20090930", "20091231",
                   "20100331", "20100630", "20100930", "20101231", "20110331", "20110630", "20110930", "20111231",
                   "20120331", "20120630", "20120930", "20121231", "20130331", "20130630", "20130930", "20131231",
                   "20140331", "20140630", "20140930", "20141231", "20150331", "20150630", "20150930", "20151231",
                   "20160331", "20160630", "20160930", "20161231", "20170331", "20170630", "20170930", "20171231",
                   "20180331", "20180630", "20180930", "20181231", "20190331", "20190630", "20190930"]  # ,"20191231"
    finance_list = ["BPS,APTURNDAYS,CASHTATIO,CATOASSET,CFPS,DEDUCTEDPROFITTOEBT,EBITDAPS,EPSBASIC,INCOMESTATEMENT_14,"
                    "INCOMESTATEMENT_13,GPMARGIN,INVTURNDAYS,BALANCESTATEMENT_94,INCOMESTATEMENT_60,INCOMESTATEMENT_61,"
                    "INCOMESTATEMENT_27,INCOMESTATEMENT_48,ORPS,ROA,ROEWA,INCOMESTATEMENT_12,INCOMESTATEMENT_11,"
                    "INCOMESTATEMENT_83,INCOMESTATEMENT_55,UNDISTRIBUTEDPS,YOYEBT,YOYEPSBASIC,YOYNI,YOYROELILUTED"]
    # TAXPAY,OTHERCINCOME,NETPROFIT重复的指标代码
    for report_date in report_list:
        print(report_date)
        data = c.sector("001004", report_date)
        stock_list = [stock_code for i, stock_code in enumerate(data.Data) if i % 2 == 0]
        data = c.css(stock_list, finance_list[0], "ReportDate=" + report_date + ",DataAdjustType=1,type=1")
        df = pd.DataFrame(data.Data).T
        df.columns = data.Indicators
        index_list = df.index.tolist()
        for code in index_list:
            print(code)
            for _ in range(100):
                try:
                    stock_data = pd.DataFrame(df.loc[code])
                    actual_date = c.css(code, "STMTACTDATE,STMTPLANDATE", "ReportDate=" + report_date)
                    stock_data["STMTACTDATE"] = actual_date.Data[code][0]
                    stock_data["STMTPLANDATE"] = actual_date.Data[code][1]
                    if not os.path.isdir(data_path + code[0:6]):
                        os.mkdir(data_path + code[0:6])
                        stock_data.to_hdf(data_path + code[0:6] + "\\" + report_date + ".h5", report_date)
                    else:
                        stock_data.to_hdf(data_path + code[0:6] + "\\" + report_date + ".h5", report_date)
                    break
                except BaseException:
                    continue
    print("首次更新完毕！")


def finance_report_add(end_date, fina_indicator=[]):
    """
    :param end_date: 财务指标更新日期
    :param fina_indicator: 新增财务指标，list
    :return:
    """
    report_list = [x[:-3] for x in os.listdir(data_path + "000001")]
    finance_list = ",".join(pd.read_hdf(data_path + "000001\\" + report_list[-1] + ".h5",
                                        report_list[-1]).index.tolist())
    # 增量更新按照单只股票进行循环
    data = c.sector("001004", end_date)
    stock_list = [stock_code for i, stock_code in enumerate(data.Data) if i % 2 == 0]
    # 先进行全市场股票增量更新
    for code in stock_list:
        report_date = datetime.strftime(parse(c.css(code, "LASTREPORTDATE", "Type=0,FormType=0").Data[code][0]),
                                        "%Y%m%d")
        if report_date in report_list:
            print(code)
            continue
        else:
            for _ in range(100):
                try:
                    data = c.css(code, finance_list, "ReportDate=" + report_date + ",DataAdjustType=1,type=1")
                    stock_data = pd.DataFrame(data.Data)
                    stock_data.index = data.Indicators
                    actual_date = c.css(code, "STMTACTDATE,STMTPLANDATE", "ReportDate=" + report_date)
                    stock_data["STMTACTDATE"] = actual_date.Data[code][0]
                    stock_data["STMTPLANDATE"] = actual_date.Data[code][1]
                    if not os.path.isdir(data_path + code[0:6]):
                        os.mkdir(data_path + code[0:6])
                        stock_data.to_hdf(data_path + code[0:6] + "\\" + report_date + ".h5", report_date)
                    else:
                        stock_data.to_hdf(data_path + code[0:6] + "\\" + report_date + ".h5", report_date)
                    break
                except BaseException:
                    continue

    # 对新加指标进行全量更新
    finance_list = pd.read_hdf(data_path + "000001\\" + report_list[-1] + ".h5", report_list[-1]).index.tolist()
    if len(fina_indicator):
        for indicator in fina_indicator:
            if indicator in finance_list:
                continue
            else:
                stock_exit = [x + ".SH" if x[0] == "6" else x + ".SZ" for x in os.listdir(data_path)]
                for stock in stock_exit:
                    report_exit = [x[:-3] for x in os.listdir(data_path + stock[0:6])]
                    for report in report_exit:
                        for _ in range(100):
                            try:
                                data = c.css(stock, indicator, "ReportDate=" + report + ",DataAdjustType=1,type=1")
                                stock_data = pd.DataFrame(data.Data, index=[0])
                                stock_data.index = data.Indicators
                                actual_date = c.css(stock, "STMTACTDATE,STMTPLANDATE", "ReportDate=" + report)
                                stock_data["STMTACTDATE"] = actual_date.Data[stock][0]
                                stock_data["STMTPLANDATE"] = actual_date.Data[stock][1]
                                # 合并已有数据
                                fina_data = pd.read_hdf(data_path + stock[0:6] + "\\" + report + ".h5", report)
                                stock_data = pd.concat([fina_data, stock_data], axis=0)
                                stock_data.to_hdf(data_path + stock[0:6] + "\\" + report + ".h5", report)
                                break
                            except BaseException:
                                continue
    print("增量更新完毕！")


if __name__ == '__main__':
    """设置下载日期"""
    pro = ts.pro_api()
    TradeDate = pd.read_csv(Pre_path + "\\TradeDate.csv")
    start_date = str(TradeDate.iloc[0, 0])
    end_date = str(TradeDate.iloc[-1, 0])
    fina_indicator = ["INVTURNRATIO", "INCOMESTATEMENT_49", "BALANCESTATEMENT_25", "BALANCESTATEMENT_93", "BALANCESTATEMENT_17",
                      "BALANCESTATEMENT_95", "BALANCESTATEMENT_96", "EBIT", "EBITDA", "FCFF", "ARTURNDAYS", "ASSETTURNRATIO",
                      "INCOMESTATEMENT_9", "INCOMESTATEMENT_10", "DEDUCTEDINCOME", "INCOMESTATEMENT_84", "INCOMESTATEMENT_20",
                      "INCOMESTATEMENT_11", "INCOMESTATEMENT_128", "INCOMESTATEMENT_16", "INCOMESTATEMENT_50", "BALANCESTATEMENT_31",
                      "BALANCESTATEMENT_39", "BALANCESTATEMENT_46", "BALANCESTATEMENT_103", "BALANCESTATEMENT_128", "BALANCESTATEMENT_74",
                      "BALANCESTATEMENT_132", "BALANCESTATEMENT_131", "BALANCESTATEMENT_140", "BALANCESTATEMENT_141", "FCFFPS", "NPMARGIN",
                      "LIBILITYTOASSET", "ASSETSTOEQUITY"]
    finance_report_add(end_date, fina_indicator)
