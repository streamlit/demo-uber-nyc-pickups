# -*- coding: utf-8 -*-
# Copyright 2018-2019 Streamlit Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""An example of showing geographic data."""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

import sys

DATE_TIME = "date/time"
DATA_URL = (
    "http://s3-us-west-2.amazonaws.com/streamlit-demo-data/uber-raw-data-sep14.csv.gz"
)

st.title("Uber Pickups in New York City")
st.markdown(
    (
        "This is a demo of a Streamlit app that shows the Uber pickups"
        " time and space distribution in New York City. Use the slider in the sidebar to the"
        " left to pick a specific hour and look at the charts below."
    )
)


@st.cache(persist=True)
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis="columns", inplace=True)
    data[DATE_TIME] = pd.to_datetime(data[DATE_TIME])
    return data


data = load_data(100000)

hour = st.sidebar.slider("Hour to look at", 0, 23)

st.sidebar.subheader("Pickups by hour")

chart_data = pd.DataFrame(
    {
        "hour": range(24),
        "pickups": np.histogram(data[DATE_TIME].dt.hour, bins=24, range=(0, 24))[0],
    }
)

chart = (
    alt.Chart(chart_data, height=120)
    .mark_area()
    .encode(alt.X("hour:Q", scale=alt.Scale(nice=False)), alt.Y("pickups:Q"))
)
selected_frame_df = pd.DataFrame({"selected_frame": [hour]})
vline = (
    alt.Chart(selected_frame_df)
    .mark_rule(color="red")
    .encode(alt.X("selected_frame:Q", axis=None))
)
st.sidebar.altair_chart(alt.layer(chart, vline))

st.sidebar.markdown(
    "[See source code](https://github.com/streamlit/streamlit/blob/develop/examples/uber.py)"
)

data = data[data[DATE_TIME].dt.hour == hour]

st.subheader("Geo data between %i:00 and %i:00" % (hour, (hour + 1) % 24))
midpoint = (np.average(data["lat"]), np.average(data["lon"]))
st.deck_gl_chart(
    viewport={
        "latitude": midpoint[0],
        "longitude": midpoint[1],
        "zoom": 11,
        "pitch": 50,
    },
    layers=[
        {
            "type": "HexagonLayer",
            "data": data,
            "radius": 100,
            "elevationScale": 4,
            "elevationRange": [0, 1000],
            "pickable": True,
            "extruded": True,
        }
    ],
)

st.subheader("Breakdown by minute between %i:00 and %i:00" % (hour, (hour + 1) % 24))
filtered = data[
    (data[DATE_TIME].dt.hour >= hour) & (data[DATE_TIME].dt.hour < (hour + 1))
]
hist = np.histogram(filtered[DATE_TIME].dt.minute, bins=60, range=(0, 60))[0]
chart_data = pd.DataFrame(
    {
        "minute": range(60),
        "pickups": np.histogram(filtered[DATE_TIME].dt.minute, bins=60, range=(0, 60))[
            0
        ],
    }
)
chart = (
    alt.Chart(chart_data, height=120)
    .mark_bar()
    .encode(alt.X("minute:Q", scale=alt.Scale(nice=False)), alt.Y("pickups:Q"))
)
st.altair_chart(chart)

st.markdown("---")
if st.checkbox("Show raw data", False):
    st.subheader("Raw data")
    st.write(data)
